#ifndef _VOX_HPP
#define _VOX_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> avg_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution);

at::Tensor avg_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt);

at::Tensor mpm_p2g_forward(
  const at::Tensor coords,
  const at::Tensor features,
  const at::Tensor batch_index,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx);

std::vector<at::Tensor>  mpm_p2g_backward(
  const at::Tensor coords,
  const at::Tensor features,
  const at::Tensor batch_index,
  const at::Tensor voxel_grad,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx);

at::Tensor mpm_g2p_forward(
  const at::Tensor coords,
  const at::Tensor voxels,
  const at::Tensor batch_index,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx);

std::vector<at::Tensor>  mpm_g2p_backward(
  const at::Tensor coords,
  const at::Tensor voxels,
  const at::Tensor batch_index,
  const at::Tensor feature_grad,
  const int gx, const int gy, const int gz,
  const int batch_size, float dx);
#endif