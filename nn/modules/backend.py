import os

from torch.utils.cpp_extension import load
_src_path = os.path.dirname(os.path.abspath(__file__))


lib = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'voxel.cu',
                    'voxel.cpp',
                    'bindings.cpp',
                ]]
           )