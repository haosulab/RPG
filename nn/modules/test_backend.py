import numpy as np
from solver.modules.backend import lib


def random_coord(n):
    a = np.random.random()
    b = np.random.random()
    if a > b:
        a, b = b, a
    a = max(min(a, 0.8), 0.05)
    b = min(max(a+0.1, b), 0.9)
    return np.random.random((n,)) * (b-a) + a

def random_points(n):
    return np.stack((random_coord(n), random_coord(n), random_coord(n)), axis=1)

def generate_one(env, n=2000, f=10):
    x = random_points(n)
    f = np.random.normal(size=(n, f))

    outs = []
    voxel_grad = []
    x_grad = []

    g2v = []
    f_grads = []
    f_voxel_grad = []
    f_x_grad = []
    for i in range(f.shape[1]):
        for j in range(1):
            # env.simulator.states[0].clear_grad(env.simulator.stream0)
            env.simulator.particle_mass.upload(f[:, i])
            grid = env.simulator.compute_grid_mass(x)
            if j == 0:
                grid_backward = np.random.normal(size=grid.shape)
            outs.append(grid)

            env.simulator.temp.clear_grad(env.simulator.stream0)
            import torch
            env.simulator.compute_grid_mass(x, backward_grad=torch.tensor(grid_backward))
            voxel_grad.append(grid_backward)
            env.simulator.sync()
            # print(env.simulator.states[-1].x_grad.download(4))

            state = env.simulator.states[-1]
            state1 = env.simulator.states[-2]
            x_grad.append(state.x_grad.download(n))

            
            # p2g


            env.simulator.temp.grid_v_out.upload(np.stack((grid, grid, grid), -1).reshape(-1, 3), strict=True) # all the same..
            state.clear_grad(env.simulator.stream0)
            state1.clear_grad(env.simulator.stream0)

            env.simulator.n_particles = n
            env.simulator.g2p(state, env.simulator.temp, state1)
            g2v.append(state1.v.download(n)[:, 0])

            f_grad = np.random.normal(size=g2v[-1].shape)
            f_grads.append(f_grad)

            ff = np.stack((f_grad, f_grad*0, f_grad*0), -1)
            state1.v_grad.upload(ff)
            env.simulator.g2p_grad(state, env.simulator.temp, state1)
            f_voxel_grad.append(env.simulator.temp.grid_v_out_grad.download().reshape(64, 64, 64, 3)[:, :, :, 0])
            f_x_grad.append(state.x_grad.download(n))

    return {
        'voxel': np.stack(outs, 0),
        'x': x,
        'f': f,

        'x_grad': np.stack(x_grad, 1),
        'voxel_grad': np.stack(voxel_grad, 0),

        'g2v': np.stack(g2v, -1),
        'f_grad': np.stack(f_grads, -1),
        'f_voxel_grad': np.stack(f_voxel_grad, 0),
        'f_x_grad': np.stack(f_x_grad, -1),
    }

def generate_batch(env, batch_size=32, f=10, n_range=(100, 400)):
    x = []
    fea = []
    voxels = []
    ind = []
    voxel_grad = []
    x_grad = []

    g2v = []
    f_grads = []
    f_voxel_grad=[]
    f_x_grad=[]
    for i in range(batch_size):
        n = np.random.randint(*n_range)
        tmp = generate_one(env, n, f)
        x.append(tmp['x'])
        x_grad.append(tmp['x_grad'])
        fea.append(tmp['f'])
        ind.append(np.zeros(n, dtype=np.int32)+i)
        voxels.append(tmp['voxel'])
        voxel_grad.append(tmp['voxel_grad'])
        g2v.append(tmp['g2v'])

        f_grads.append(tmp['f_grad'])
        f_voxel_grad.append(tmp['f_voxel_grad'])
        f_x_grad.append(tmp['f_x_grad'])

    return {
        'x': np.concatenate(x), 
        'fea': np.concatenate(fea), 
        'ind': np.concatenate(ind), 
        'voxel': np.stack(voxels),
        'voxel_grad': np.stack(voxel_grad),
        'x_grad': np.concatenate(x_grad),
        'g2v': np.concatenate(g2v),
        'f_grads': np.concatenate(f_grads),
        'f_voxel_grad': np.stack(f_voxel_grad),
        'f_x_grad': np.concatenate(f_x_grad),
    }