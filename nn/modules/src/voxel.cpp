#include "voxel.hpp"
#include "voxel.cuh"

#include "utils.h"

/*
  Function: average pool voxelization (forward)
  Args:
    features: features, FloatTensor[b, c, n]
    coords  : coords of each point, IntTensor[b, 3, n]
    resolution : voxel resolution
  Return:
    out : outputs, FloatTensor[b, c, s], s = r ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
*/
std::vector<at::Tensor> avg_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution) {
  CHECK_CUDA_v2(features);
  CHECK_CUDA_v2(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int r = resolution;
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor ind = torch::zeros(
      {b, n}, at::device(features.device()).dtype(at::ScalarType::Int));
  at::Tensor out = torch::zeros(
      {b, c, r3}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor cnt = torch::zeros(
      {b, r3}, at::device(features.device()).dtype(at::ScalarType::Int));
  avg_voxelize(b, c, n, r, r2, r3, coords.data_ptr<int>(),
               features.data_ptr<float>(), ind.data_ptr<int>(),
               cnt.data_ptr<int>(), out.data_ptr<float>());
  return {out, ind, cnt};
}

/*
  Function: average pool voxelization (backward)
  Args:
    grad_y : grad outputs, FloatTensor[b, c, s]
    indices: voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
  Return:
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
at::Tensor avg_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt) {
  CHECK_CUDA_v2(grad_y);
  CHECK_CUDA_v2(indices);
  CHECK_CUDA_v2(cnt);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(cnt);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_INT(cnt);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int s = grad_y.size(2);
  int n = indices.size(1);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  avg_voxelize_grad(b, c, n, s, indices.data_ptr<int>(), cnt.data_ptr<int>(),
                    grad_y.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}



//std::vector<at::Tensor> 
at::Tensor mpm_p2g_forward(
  const at::Tensor coords,
  const at::Tensor features,
  const at::Tensor batch_index,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx) {
  CHECK_CUDA_v2(features);
  CHECK_CUDA_v2(coords);
  CHECK_CUDA_v2(batch_index);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_CONTIGUOUS(batch_index);
  CHECK_IS_FLOAT(features);
  CHECK_IS_FLOAT(coords);
  CHECK_IS_INT(batch_index);

  int b = features.size(0);
  int c = features.size(1);
  at::Tensor voxel = torch::zeros(
      {batch_size, c, gx, gy, gz}, at::device(features.device()).dtype(at::ScalarType::Float));

  mpm_point2voxel(coords.data_ptr<float>(), features.data_ptr<float>(), gx, gy, gz, dx, voxel.data_ptr<float>(), batch_index.data_ptr<int>(), 0, c, b);
  //return {out, ind, cnt};
  return voxel;
}

std::vector<at::Tensor>  mpm_p2g_backward(
  const at::Tensor coords,
  const at::Tensor features,
  const at::Tensor batch_index,
  const at::Tensor voxel_grad,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx) {
  CHECK_CUDA_v2(features);
  CHECK_CUDA_v2(coords);
  CHECK_CUDA_v2(batch_index);
  CHECK_CUDA_v2(voxel_grad);

  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_CONTIGUOUS(batch_index);
  CHECK_CONTIGUOUS(voxel_grad);

  CHECK_IS_FLOAT(features);
  CHECK_IS_FLOAT(coords);
  CHECK_IS_INT(batch_index);
  CHECK_IS_FLOAT(voxel_grad);

  int b = features.size(0);
  int c = features.size(1);
  at::Tensor coord_grad = torch::zeros(
      {b, 3}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor feature_grad = torch::zeros(
      {b, c}, at::device(features.device()).dtype(at::ScalarType::Float));
  mpm_point2voxel_grad(coords.data_ptr<float>(), features.data_ptr<float>(), gx, gy, gz, dx, 0, voxel_grad.data_ptr<float>(), coord_grad.data_ptr<float>(), feature_grad.data_ptr<float>(), batch_index.data_ptr<int>(), 0, c, b);
  //return {out, ind, cnt};
  return {coord_grad, feature_grad};
}


at::Tensor mpm_g2p_forward(
  const at::Tensor coords,
  const at::Tensor voxels,
  const at::Tensor batch_index,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx) {
  CHECK_CUDA_v2(voxels);
  CHECK_CUDA_v2(coords);
  CHECK_CUDA_v2(batch_index);
  CHECK_CONTIGUOUS(voxels);
  CHECK_CONTIGUOUS(coords);
  CHECK_CONTIGUOUS(batch_index);
  CHECK_IS_FLOAT(voxels);
  CHECK_IS_FLOAT(coords);
  CHECK_IS_INT(batch_index);

  int b = coords.size(0);
  int c = voxels.size(1);
  at::Tensor features = torch::zeros(
      {b, c}, at::device(coords.device()).dtype(at::ScalarType::Float));
  mpm_point2voxel(coords.data_ptr<float>(), features.data_ptr<float>(), gx, gy, gz, dx, voxels.data_ptr<float>(), batch_index.data_ptr<int>(), 1, c, b);
  return features;
}


std::vector<at::Tensor>  mpm_g2p_backward(
  const at::Tensor coords,
  const at::Tensor voxels,
  const at::Tensor batch_index,
  const at::Tensor feature_grad,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx) {
  CHECK_CUDA_v2(voxels);
  CHECK_CUDA_v2(coords);
  CHECK_CUDA_v2(batch_index);
  CHECK_CUDA_v2(feature_grad);

  CHECK_CONTIGUOUS(voxels);
  CHECK_CONTIGUOUS(coords);
  CHECK_CONTIGUOUS(batch_index);
  CHECK_CONTIGUOUS(feature_grad);

  CHECK_IS_FLOAT(voxels);
  CHECK_IS_FLOAT(coords);
  CHECK_IS_INT(batch_index);
  CHECK_IS_FLOAT(feature_grad);

  int b = feature_grad.size(0);
  int c = feature_grad.size(1);
  at::Tensor coord_grad = torch::zeros(
      {b, 3}, at::device(feature_grad.device()).dtype(at::ScalarType::Float));
  at::Tensor voxel_grad = torch::zeros(
      {batch_size, c, gx, gy, gz}, at::device(feature_grad.device()).dtype(at::ScalarType::Float));
  mpm_point2voxel_grad(coords.data_ptr<float>(), 0, gx, gy, gz, dx, voxels.data_ptr<float>(), voxel_grad.data_ptr<float>(), coord_grad.data_ptr<float>(), feature_grad.data_ptr<float>(), batch_index.data_ptr<int>(), 1, c, b);
  //return {out, ind, cnt};
  return {coord_grad, voxel_grad};
}