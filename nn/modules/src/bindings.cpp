#include <pybind11/pybind11.h>
#include "voxel.hpp"

PYBIND11_MODULE(_pvcnn_backend, m) {
  m.def("avg_voxelize_forward", &avg_voxelize_forward,
        "Voxelization forward with average pooling (CUDA)");
  m.def("avg_voxelize_backward", &avg_voxelize_backward,
        "Voxelization backward (CUDA)");

  m.def("mpm_p2g_forward", &mpm_p2g_forward,
        "Voxelization forward with MPM (CUDA)");
  m.def("mpm_p2g_backward", &mpm_p2g_backward,
        "Voxelization backward with MPM (CUDA)");

  m.def("mpm_g2p_forward", &mpm_g2p_forward,
        "Interpolation forward with MPM (CUDA)");
  m.def("mpm_g2p_backward", &mpm_g2p_backward,
        "Interpolation backward with MPM (CUDA)");
}