#include "utils.h"
// https://github.com/mit-han-lab/pvcnn/blob/476715b113f9c119f88003729bc6475285f78819/modules/functional/src/voxelization/vox.cpp
#include <assert.h>
#include <cstdio>

#include <curand.h>
#include <curand_kernel.h>

#include "cuda_utils.cuh"
#include "../../../mpm/csrc/vec3.h"
#include "../../../mpm/csrc/mat3.h"
#include "../../../mpm/csrc/common.h"


using namespace maniskill;

// feature2voxel
__global__ void
mpm_point2voxel_kernel(
    float *__restrict__ xyz, //n x 3
    float *__restrict__ feature, //n x c 
    ivec3 grid_dim, // dx, dy, dz

    float dx,
    float inv_dx,
    float *__restrict__ voxel,
    int *__restrict__ batch_index,
    int direction, //0 p2g 1 g2p
    int c,
    int dim)
{
    int p = get_tid();
    if (p < dim)
    {
        xyz = xyz + p * 3;
        int batch_id = batch_index[p];
        feature = feature + p * c;

        int stride = grid_dim.x * grid_dim.y * grid_dim.z;
        voxel = voxel + batch_id * c * stride;

        vec3 x(xyz[0], xyz[1], xyz[2]);
        ivec3 base = cast_int(x * inv_dx - 0.5);
        vec3 fx = x * inv_dx - cast_float(base);
        mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f), 0.5f * pow2(fx - 0.5f));


        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                {
                    float weight = w(0, i) * w(1, j) * w(2, k);
                    vec3 dpos = (vec3(i, j, k) - fx) * dx;
                    ivec3 i3 = base + ivec3(i, j, k);
                    //printf("%d %f %f %f %d %d %d %f %d\n", p, xyz[0], xyz[1], xyz[2], i3.x, i3.y, i3.z, feature[0], c);
                    int index = grid_index(i3, grid_dim);
                    for (int l = 0; l < c; ++l){
                      if(direction == 0){
                        ::atomicAdd(&voxel[l * stride + index], feature[l] * weight);
                      }else{
                        feature[l] += voxel[l * stride + index] * weight;
                      }
                    }
                }
    }
}


inline CUDA_CALLABLE vec3 dw(mat3 const &weights0, mat3 const &weights1, const int &i, const int &j, const int &k)
{
  // https://github.com/yuanming-hu/ChainQueen/blob/0bdda869d66b483dc85b8966e4d5f2b8200021e9/src/state.cuh#L461
  return vec3(
      weights1.data[0][i] * weights0.data[1][j] * weights0.data[2][k],
      weights0.data[0][i] * weights1.data[1][j] * weights0.data[2][k],
      weights0.data[0][i] * weights0.data[1][j] * weights1.data[2][k]);
}

// feature2voxel
__global__ void
mpm_point2voxel_grad_kernel(
    float *__restrict__ xyz, //n x 3
    float *__restrict__ feature, //n x c 
    ivec3 grid_dim, // dx, dy, dz

    float dx,
    float inv_dx,
    float *__restrict__ voxel,
    float *__restrict__ voxel_grad,
    float *__restrict__ x_grad,
    float *__restrict__ feature_grad,

    int *__restrict__ batch_index,
    int direction,
    int c,
    int dim)
{
    int p = get_tid();
    if (p < dim)
    {
        xyz += p*3;
        x_grad += p*3;
        int batch_id = batch_index[p];
        int stride = grid_dim.x * grid_dim.y * grid_dim.z;

        feature += p*c;
        voxel += batch_id * c * stride;

        feature_grad += p*c;
        voxel_grad += batch_id * c * stride;

        vec3 x(xyz[0], xyz[1], xyz[2]);
        ivec3 base = cast_int(x * inv_dx - 0.5);
        vec3 fx = x * inv_dx - cast_float(base);
        mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f), 0.5f * pow2(fx - 0.5f));
        mat3 w1 = mat3(-inv_dx * (1.5f - fx), inv_dx * ((-2.f) * fx + 2.0f), -inv_dx * (fx * (-1.f) + 0.5f));
        vec3 grad_x = vec3(0.);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                {
                    ivec3 i3 = base + ivec3(i, j, k);
                    int index = grid_index(i3, grid_dim);
                    float weight = w(0, i) * w(1, j) * w(2, k);
                    auto grad_N = dw(w, w1, i, j, k); // magic function from chainqueen's code
                    for (int l = 0; l < c; ++l)
                    {
                      /*
                      if(direction == 0){
                        ::atomicAdd(&voxel[index * c + l], feature[l] * weight);
                      }else{
                        feature[l] += voxel[index * c + l] * weight;
                      }
                      */
                      if(direction==0){
                        auto grad_grid = voxel_grad[l*stride + index];
                        feature_grad[l] += weight * grad_grid;
                        grad_x += grad_grid * feature[l] * grad_N; // mN term
                      } else {
                        auto grad_feature = feature_grad[l];
                        ::atomicAdd(&voxel_grad[index + l * stride], grad_feature * weight);
                        grad_x += grad_feature * voxel[index + l * stride] * grad_N; // mN term
                      }
                    }
                }
        x_grad[0] += grad_x.x;
        x_grad[1] += grad_x.y;
        x_grad[2] += grad_x.z;
    }
}

#define launch_kernel_v2(kernel, dim, args)                               \
  {                                                                            \
    const int num_threads = 256;                                               \
    const int num_blocks = (dim + num_threads - 1) / num_threads;              \
    kernel<<<num_blocks, 256, 0>>> args;                               \
  }


void mpm_point2voxel(
    float *__restrict__ xyz, //n x 3
    float *__restrict__ feature, //n x c 
    int gx, int gy, int gz,
    float dx,
    float *__restrict__ voxel,
    int *__restrict__ batch_index,
    int d,
    int c,
    int dim){
    ivec3 grid_dim(gx, gy, gz);
    launch_kernel_v2(mpm_point2voxel_kernel, dim, (xyz, feature, grid_dim, dx, 1./dx, voxel, batch_index, d, c, dim));
}

void mpm_point2voxel_grad(
    float *__restrict__ xyz, //n x 3
    float *__restrict__ feature, //n x c 
    int gx, int gy, int gz,
    float dx,
    float *__restrict__ voxel,
    float *__restrict__ voxel_grad,
    float *__restrict__ x_grad,
    float *__restrict__ feature_grad,
    int *__restrict__ batch_index,
    int d,
    int c,
    int dim){
    ivec3 grid_dim(gx, gy, gz);
    launch_kernel_v2(mpm_point2voxel_grad_kernel, dim, (xyz, feature, grid_dim, dx, 1./dx, voxel, voxel_grad, x_grad, feature_grad, batch_index, d, c, dim));
}


/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, IntTensor[b, 3, n]
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
*/
__global__ void grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const int *__restrict__ coords,
                                  int *__restrict__ ind, int *cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  ind += batch_index * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    // if (ind[i] == -1)
    //   continue;
    ind[i] = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    atomicAdd(cnt + ind[i], 1);
  }
}

/*
  Function: average pool voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void avg_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(out + j * s + pos, feat[j * n + i] * div_cur_cnt);
      }
    }
  }
}

/*
  Function: average pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void avg_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ ind,
                                         const int *__restrict__ cnt,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  cnt += batch_index * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * div_cur_cnt);
      }
    }
  }
}

void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *cnt, float *out) {
  grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, coords, ind,
                                                   cnt);
  avg_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, ind, cnt,
                                                     feat, out);
  CUDA_CHECK_ERRORS();
}

void avg_voxelize_grad(int b, int c, int n, int s, const int *ind,
                       const int *cnt, const float *grad_y, float *grad_x) {
  avg_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, ind, cnt,
                                                          grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}


