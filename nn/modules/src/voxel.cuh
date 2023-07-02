#ifndef _VOX_CUH
#define _VOX_CUH

// CUDA function declarations
void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *cnt, float *out);
void avg_voxelize_grad(int b, int c, int n, int s, const int *idx,
                       const int *cnt, const float *grad_y, float *grad_x);


void mpm_point2voxel(
    float *__restrict__ xyz, //n x 3
    float *__restrict__ feature, //n x c 
    int gx, int gy, int gz, // dx, dy, dz
    float dx,
    float *__restrict__ voxel,
    int *__restrict__ batch_index,
    int d,
    int c,
    int dim);

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
    int dim);
#endif