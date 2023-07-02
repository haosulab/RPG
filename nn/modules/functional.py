import torch
from torch.autograd import Function

from .backend import lib

__all__ = ['avg_voxelize', 'mpm_p2g', 'mpm_g2p']


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = lib.avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = lib.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


avg_voxelize = AvgVoxelization.apply

def check_coords(coords, dx, resolution):
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)

    min_x = coords.min(axis=0)[0]
    max_x = coords.max(axis=0)[0]

    assert (min_x >= dx * 0.5).all()
    upper = ((max_x/dx) + 2 - 0.5)
    assert upper[0] < resolution[0] and upper[1] < resolution[1] and upper[2] < resolution[2], f"{upper} {resolution}"

    assert len(resolution) == 3
    return coords, resolution

class MPMP2G(Function):
    @staticmethod
    def forward(ctx, coords, features, resolution, batch_index, dx):
        coords, resolution = check_coords(coords, dx, resolution)
        if isinstance(batch_index, torch.LongTensor):
            batch_index = batch_index.int()

        features = features.contiguous()
        coords = coords.contiguous()
        _, c = features.shape
        batch_size = int(batch_index.max() + 1)
        ctx.dx = dx
        ctx.save_for_backward(coords, features, batch_index)

        out = lib.mpm_p2g_forward(coords, features, batch_index, *resolution, batch_size, dx)
        return out.view(batch_size, c, *resolution)

    @staticmethod
    def backward(ctx, grad_output):
        batch_size = grad_output.shape[0]
        coords, features, batch_index = ctx.saved_tensors
        dx = ctx.dx

        grad_output = grad_output.contiguous()
        # print(grad_output.shape)
        resolution = grad_output.shape[2:5]
        x_grad, f_grad = lib.mpm_p2g_backward(coords, features, batch_index, grad_output, *resolution, batch_size, dx)
        return x_grad, f_grad, None, None, None

mpm_p2g = MPMP2G.apply


class MPMG2P(Function):
    @staticmethod
    def forward(ctx, coords, voxels, batch_index, dx):
        # TODO: check resolution
        if isinstance(batch_index, torch.LongTensor):
            batch_index = batch_index.int()

        resolution = voxels.shape[2:]
        coords, _ = check_coords(coords, dx, voxels.shape[2:])
        voxels = voxels.contiguous()
        coords = coords.contiguous()
        batch_size = int(batch_index.max() + 1)
        ctx.dx = dx
        ctx.save_for_backward(coords, voxels, batch_index)
        out = lib.mpm_g2p_forward(coords, voxels, batch_index, *resolution, batch_size, dx)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        coords, voxels, batch_index = ctx.saved_tensors
        batch_size = voxels.shape[0]
        dx = ctx.dx
        resolution = voxels.shape[2:5]
        x_grad, voxel_grad = lib.mpm_g2p_backward(coords, voxels, batch_index, grad_output, *resolution, batch_size, dx)
        return x_grad, voxel_grad, None, None, None

mpm_g2p = MPMG2P.apply