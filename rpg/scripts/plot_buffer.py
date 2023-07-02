import torch
import matplotlib.pyplot as plt
from rpg.scripts.plot import create_axes
import numpy as np
from solver.draw_utils import plot_colored_embedding


def main(name):
    buffer = torch.load(f'data/savedbuffer/{name}/buffer.pt')
    #print(buffer.total_size())
    total = buffer.total_size()
    obs = buffer._obs[:total]
    z = buffer._z[:total]
    #print(obs.shape, z.shape)
    
    band = 1000

    end = total
    start = band 

    bins = 5

    middlepoint = (np.arange(bins) + 1) * (total - band)//bins
    #middlepoint = np.append(0, middlepoint)
    middlepoint = [0, 20000, 40000, total - band-1]

    axes = create_axes(10, middlepoint)

    from envs.maze import CrossMaze

    maze = CrossMaze()
    background = maze.render_wall(linewidth=3, linecolor=(42, 42, 165))
    

    for ax, j in zip(axes, middlepoint):
        if 'mbsac' in name:
            band = 500
            end = min(j + band, total)
            start = end - band
            oo = obs[start:end][..., :2]
            zz = torch.zeros(oo.shape[0]).long()
        else:
            band = 500
            end = min(j + band, total)
            start = end - band
            zz = z[start:end][..., :3]
            oo = obs[start:end][..., :2]

            idx = torch.tensor(np.random.choice(min(1000, band), band)).long().cuda()
            zz = zz[idx]
            oo = oo[idx]

        #print(zz.shape, oo.shape)
        #print(zz.shape, oo.shape)
        oo = maze.pos2pixel(oo / 0.01)
        print(oo.shape, zz.shape, zz.dtype)
        ax.imshow(background)
        plot_colored_embedding(zz, oo, ax=ax)
        
    plt.tight_layout()
    plt.savefig(f'data/{name}_buffer.png', dpi=300)

    
def sample_and_plot(path, idx):
    import os
    from rpg.env_base import make
    from tools.utils import logger
    logger.configure()
    env = make('GapMaze', n=50, obs_dim=6)

    agent = torch.load(os.path.join(path, f'model{idx}.pt'))
    agent.env = env
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(6, 6))

    agent.z = None
    traj = agent.evaluate(None, 200)
    next_obs = traj.get_tensor('next_obs')
    z = traj.get_tensor('z')

    background = env.goal_env.render_wall(linewidth=3, linecolor=(42, 42, 165), background_color=(255, 255, 255))
    print(next_obs.shape)
    print(z.shape)

    oo = env.goal_env.pos2pixel(next_obs[..., :2] / 0.01)
    plt.imshow(background)
    print(type(z))

    #plot_colored_embedding(z, oo, s=2)
    oo = oo.reshape(-1, 2)
    if isinstance(oo, torch.Tensor):
        oo = oo.detach().cpu().numpy()

    if z.dtype == torch.int32 or z.dtype == torch.int64:
        z = 'b'
    else:
        z = ((z - -3.) / (3. - -3.)).clamp(0, 1.)
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        z = z[..., :3].reshape(-1, 3)
    
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.scatter(oo[:, 0], oo[:, 1], c=z, s=2)
    plt.axis('off')
    plt.tight_layout()

    method = path.split('/')[-1]
    os.makedirs(f'data/maze/{method}', exist_ok=True)

    plt.savefig(f'data/maze/{method}/{idx}_buffer.png', dpi=300)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    if args.name == 'gaussian':
        for i in range(50, 1001,50):
            print(i)
            sample_and_plot('data/buffers/gaussian_seed8', i)
            sample_and_plot('data/buffers/gaussian_seed3', i)
            sample_and_plot('data/buffers/gaussian_seed4', i)
            sample_and_plot('data/buffers/gaussian_seed5', i)
            sample_and_plot('data/buffers/gaussian_seed6', i)
            sample_and_plot('data/buffers/gaussian_seed7', i)
    else:
        for i in range(50, 1001,50):
            print(i)
            sample_and_plot('data/buffers/mbsac_seed7', i)
            sample_and_plot('data/buffers/mbsac_seed8', i)
            sample_and_plot('data/buffers/mbsac_seed3', i)
            sample_and_plot('data/buffers/mbsac_seed4', i)
            sample_and_plot('data/buffers/mbsac_seed5', i)
            sample_and_plot('data/buffers/mbsac_seed6', i)