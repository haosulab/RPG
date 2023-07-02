# %%
#%load_ext autoreload
#%autoreload 2

# %%
from utils_visualise import plot_curves, plot4figs
from matplotlib import pyplot as plt

# %%
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
default_settings = Namespace(device='cuda:0', entropy=0., plot=False, max_steps=150)
#parser.add_argument('--device', default='cuda:0', type=str)
#parser.add_argument('--entropy', default=0., type=float)
#parser.add_argument('--plot', action='store_true', default=False)
#parser.add_argument('--max_steps', type=int, default=10000)

# %%
n_run = 3
POLICYS = ['GMM-v1', 'normal-v1', 'normal-v2', 'Gumbel']
COLORS = ['green', 'blue', 'magenta', 'red']

# %%
args = Namespace(policy='normal-v1', batch_size=128, grad_size=0.6, maximum=4., init_std=0.1, **default_settings.__dict__)
#parser.add_argument('--policy', default='normal-v1', type=str, choices=['normal-v1', 'normal-v2', 'GMM-v1', 'GMM-v2', 'GMM-v3', 'Gumbel'])
#parser.add_argument('--batch_size', type=int, default=128)
#parser.add_argument('--init_std', type=float, default=0.1)

# %%
#plot_curves(args, n_run, POLICYS, COLORS)

# %%

GRAD_SIZE_LIST = [0.6, 10]
MAXIMUM_LIST = [4., 0.]
#parser.add_argument('--grad_size', type=float, default=0.6) # choices 0.6, 10.
#parser.add_argument('--maximum', type=float, default=4.) # choices 4., 0.

# %%
INIT_STD_LIST = [0.1, 0.2, 0.6]
BATCH_SIZE_LIST = [1, 128, 100000]
for init_std in INIT_STD_LIST:
    print('init-std:', init_std)
    args.init_std = init_std
    for batch_size in BATCH_SIZE_LIST:
        args.batch_size = batch_size
        if batch_size == 1:
            args.max_steps = 600
        else:
            args.max_steps = 150
        plot4figs(args, n_run, GRAD_SIZE_LIST, MAXIMUM_LIST, POLICYS, COLORS)

# %%


# %%


# %%



