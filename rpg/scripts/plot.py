import collections
import numpy as np
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas

font = {'size': 34}
import matplotlib
import os
matplotlib.rc('font', **font)

def merge_curves(x_list, y_list, bin_width=1000, max_steps=None):
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    idx = x.argsort()
    x = x[idx]
    y = y[idx]

    if max_steps is not None:
        idx = x <= max_steps
        x = x[idx]
        y = y[idx]
    assert (x >= 0).all()
    nbins = int(x.max() // bin_width + 1)
        
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    xx, _ = np.histogram(x, bins=nbins, weights=x)

    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    xx = xx / n
    idx = xx>0
    return xx[idx]/1e6, mean[idx], std[idx]


def smooth(y, smoothingWeight=0.95):
    y_smooth = []
    last = y[0]
    for i in range(len(y)):
        y_smooth.append(last * smoothingWeight + (1 - smoothingWeight) * y[i])
        last = y_smooth[-1]
    return np.array(y_smooth)


def plot_curve_with_shade(ax, x, mean, std, label, color='green', smoothingWeight=0, linewidth=3, **kwargs):
    
    #y_smooth = mean
    y_smooth = smooth(mean, smoothingWeight)
    std_smooth = smooth(std, smoothingWeight) * 0.3
    ax.fill_between(x, (y_smooth - std_smooth).clip(0, np.inf), y_smooth + std_smooth, facecolor=color, alpha=0.2)
    ax.plot(x, y_smooth, color=color, label=label, linewidth=linewidth, **kwargs)


KEYS = defaultdict(lambda: 'success')
MAX_STEPS = defaultdict(lambda: 2000000)
KEYS['densecabinet'] = 'train_door0_metric'
MAX_STEPS['densecabinet'] = 800000
MAX_STEPS['antpush'] = 1000000

MAX_STEPS['kitchen'] = 2000000
#MAX_STEPS['antfall'] = 200000
KEYS['kitchen'] = 'train_microwave_metric'


def get_df_xy(df, x, y):
    x = df[x]
    y = df[y]
    data = np.stack([x, y]).T
    data = data[~np.isnan(y)]
    return data[:, 0], data[:, 1]

def read_baseline_result(env_name, method):
    ENVS = dict(
        densecabinet='CabinetDense',
        antpush='AntPushDense',
        door='AdroitDoor',
        hammer='AdroitHammer',
        stickpull='MWStickPull',
        basket='MWBasketBall',
        cabinet='Cabinet',
        block='BlockPush2',
        kitchen='Kitchen4',
    )
    if env_name not in ENVS:
        return [], []
    env_name = ENVS[env_name]

    x_keys = dict(sac='env_n_samples', sac_no_expl='env_n_samples', tdmpc='env_steps', dreamer_p2e='env_n_samples')

    if not method.startswith('sac'):
        query = f"data/collected_results/{method}/{env_name}/*/progress.csv"
    else:
        query = f"data/collected_results/{method}/{env_name}/*/*/progress.csv"
    xs = []
    ys = []
    for path in list(glob.iglob(query)):
        try:
            df = pandas.read_csv(path)
        except Exception as e:
            # print(e)
            continue

        key = 'eval_success_rate'
        if method in ['sac', 'tdmpc', 'sac_no_expl']:
            # df = df[df['eval_success_rate'] > 0]
            if env_name in ["Kitchen4", "Kitchen2"]: # , 
                key = "eval_metrics_true"
            elif env_name in ["CabinetDense"]:
                key = "eval_metrics_door0"
        if method.startswith('sac'):
            key = key.replace('eval', 'train')

        x, y = get_df_xy(df, x_keys[method], key)
        # print(path, len(x))
        if len(x) == 0:
            print(method, env_name, len(x), key)
            print(path)
            print(pandas.read_csv(path))
            exit(0)
        xs.append(x); ys.append(y)

    
    #print(env_name, method)
    if env_name == 'AdroitDoor':
        if method == 'dreamer_p2e':
            print('+' * 10, env_name, method, [len(i) for i in xs])

            idx = [i for i in range(len(xs)) if len(xs[i]) > 130]
            xs = [xs[i] for i in idx]
            ys = [ys[i] for i in idx]
            print([len(i) for i in xs])


    if len(xs) == 0:
        return xs, ys
    if method == 'sac_no_expl':
        for i in range(len(xs)):
            if len(xs[i]) < 400:
                xs[i] = np.arange(5000, 2000001, 5000)
                ys[i] = np.zeros(400)
    if env_name == 'BlockPush2' and method == 'dreamer_p2e':
        ys[3] = np.concatenate([ys[3], ys[0][len(xs[3]):] * 0])
        xs[3] = np.concatenate([xs[3], xs[0][len(xs[3]):]])
    elif method == 'dreamer_p2e':
        for i in range(len(xs)):
            if len(xs[i]) < 200:
                xs[i] = np.arange(5000, 2000001, 5000)
                ys[i] = np.zeros(400)
        

    if method == 'sac' and env_name == 'MWBasketBall':
        xs = [xs[0], xs[1], xs[2], xs[3], xs[5]]
        ys = [ys[0], ys[1], ys[2], ys[3], ys[5]]

    min_len = min([len(i) for i in xs])
    if env_name != 'antpush':
        xs = [i[:min_len] for i in xs]
        ys = [i[:min_len] for i in ys]
        
    return xs, ys

def read_wandb_result(env_name, method):
    xs, ys = [], []
    for path in list(glob.iglob(f"data/wandb/{env_name}/{method}/*.csv")):
        try:
            df = pandas.read_csv(path)
        except Exception as e:
            # print(e)
            continue

        x, y = get_df_xy(df, 'total_steps', KEYS[env_name])
        # print(path, len(x))
        xs.append(x); ys.append(y)
    return xs, ys


linetypes = defaultdict(lambda: '-')
linetypes['mbsac'] = '--'
linetypes['tdmpc'] = '--'
linetypes['sac'] = '--'
linetypes['MBSAC(R)'] = '--'
linetypes['TDMPC(R)'] = '--'
linetypes['SAC(R)'] = '--'

linetypes['MBSAC'] = '--'
linetypes['TDMPC'] = '--'
linetypes['SAC'] = '--'

def plot_env(ax: plt.Axes, env_name, index):
    results = dict(
        rpg=read_wandb_result(env_name, 'rpg'),
        mbsac=read_wandb_result(env_name, 'mbsac'),
        sac=read_baseline_result(env_name, 'sac'),
        tdmpc=read_baseline_result(env_name, 'tdmpc'),
        p2e=read_baseline_result(env_name, 'dreamer_p2e'),
        sac_no_expl=read_baseline_result(env_name, 'sac_no_expl')
    )
    
    #print(len(results['sac'][0]))
    COLORS = ['C3', 'C0', 'C1', 'C2', 'C4', 'C5']
    idx = 0
    
    avgs = {}
    for k, v in results.items():
        if len(v[0]) > 0:
            print(env_name, k, len(v[0]), [len(x) for x in v[0]])
            x, mean, std = merge_curves(v[0], v[1], 10000, MAX_STEPS[env_name])
            if env_name == 'block':
                mean = mean/2
                std = std/2

            avgs[k] = mean[-1]
            c = COLORS[idx]
            if k == 'sac' and env_name in ['CabinetDense', 'AntPush']:
                c = 'C5'
            plot_curve_with_shade(ax, x, mean, std,
                                label=k + f" ({len(v[0])} runs)", smoothingWeight=0.6, color = c, linestyle=linetypes[k])
        idx += 1
        
    
    # ax.legend(loc=2)
    ax.set_title("("+index +") " + env_name.capitalize().replace("Basket", "BasketBall").replace('Densecabinet', 'Cabinet(Dense)'))
    ax.set_xlabel("interactions (M)"); 
    ax.set_ylabel('Success Rate')
    ax.grid()
    return avgs
    

def create_axes(T, envs):
    width = min(T, len(envs))
    n_rows = (len(envs) + width - 1)//width
    fig, axs = plt.subplots(n_rows, width, figsize=(8 * width, 8 * n_rows))
    
    if isinstance(axs, np.ndarray):
        if isinstance(axs[0], np.ndarray):
            axs = sum([list(x) for x in axs], [])
    else:
        axs = [axs]
    return axs

def legends():
    plt.figure(figsize=(18,0.8))
    for method, color in zip(
        ['Our Method', 'MBSAC(R)', 'SAC(R)', 'TDMPC(R)', 'DreamerV2(P2E)', 'SAC'],
        ['C3', 'C0', 'C1', 'C2', 'C4', 'C5']
    ):
        plt.plot(0, 0, label=method, color=color, linestyle=linetypes[method])

    # plt.plot(0, 0, label="Ours (no pathwise grad)", color=METHOD_INFO["rrf"]["color"])
    # plt.plot(0, 0, label="Ours (no encoder)", color=METHOD_INFO["rpgnoz"]["color"])
    # plt.plot(0, 0, label="Ours (no info loss)", color=)
    # plt.plot(0, 0, label="PPO", color=METHOD_INFO["ppo"]["color"])
    # plt.plot(0, 0, label="SAC", color=METHOD_INFO["sac"]["color"])
    plt.axis('off')
    leg = plt.legend(loc='center', prop={'size': 20}, ncol=8)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    plt.savefig("sparse_lengends.png")
    plt.show()
    
if __name__ == '__main__':

    font = {'size': 20}
    import matplotlib
    import os
    matplotlib.rc('font', **font)

    # 'densecabinet', 'antpush', 

    envs = ['hammer', 'door', 'basket', 'block', 'cabinet', 'stickpull'] #, 'kitchen'] #, 'antfall']

    width = min(3, len(envs))
    n_rows = (len(envs) + width - 1)//width
    
    fig, axs = plt.subplots(n_rows, width, figsize=(8 * width, 6 * n_rows))

    averages = {}
    
    if isinstance(axs[0], np.ndarray):
        axs = sum([list(x) for x in axs], [])
    id = ord('A')
    for ax, env_name in zip(axs, envs):
        out = plot_env(ax, env_name, chr(id))
        for k, v in out.items():
            if k not in averages:
                averages[k] = []
            averages[k].append(v)
        id += 1

    averages = {k: np.mean(v) for k, v in averages.items()}

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()

    #fig.legend(handles, ['RPG', 'MBSAC(R)', 'SAC(R)', 'TDMPC(R)', 'P2E'], loc='upper center', ncol=6) # 
    #fig.subplots_adjust(top=0.95, left=0.155, right=0.99, bottom=0.2)

    # , ncol=6, prop={'size': 30}, bbox_to_anchor=(0.45, 0.12)

    # plt.tight_layout()
    plt.savefig('sparse.png', dpi=300)

    plt.clf()

    envs = ['densecabinet', 'antpush']

    width = min(1, len(envs))
    n_rows = (len(envs) + width - 1)//width
    
    fig, axs = plt.subplots(n_rows, width, figsize=(8 * width, 6 * n_rows))
    
    if isinstance(axs[0], np.ndarray):
        axs = sum([list(x) for x in axs], [])
    id = ord('A')
    for ax, env_name in zip(axs, envs):
        plot_env(ax, env_name, chr(id))
        id += 1

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    plt.savefig('dense.png', dpi=300)

    plt.clf()
    legends()


    plt.figure(figsize=(18,0.8))
    for method, color in zip(['RPG', 'MBSAC', 'SAC', 'TDMPC', 'Dreamer'], ['C3', 'C0', 'C1', 'C2', 'C4']):
        plt.plot(0, 0, label=method, color=color, linestyle=linetypes[method])
    # plt.plot(0, 0, label="Ours (no pathwise grad)", color=METHOD_INFO["rrf"]["color"])
    # plt.plot(0, 0, label="Ours (no encoder)", color=METHOD_INFO["rpgnoz"]["color"])
    # plt.plot(0, 0, label="Ours (no info loss)", color=)
    # plt.plot(0, 0, label="PPO", color=METHOD_INFO["ppo"]["color"])
    # plt.plot(0, 0, label="SAC", color=METHOD_INFO["sac"]["color"])
    plt.axis('off')
    leg = plt.legend(loc='center', prop={'size': 20}, ncol=8)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    plt.savefig("dense_lengends.png")
    plt.show()


    plt.clf()
    plt.figure(figsize=(8, 6))
    print(averages)
    KEYS = dict(mbsac='MBSAC(R)', sac='SAC(R)', tdmpc='TDMPC(R)', p2e='P2E', rpg='Ours')
    order = ['p2e', 'tdmpc', 'mbsac', 'rpg']
    plt.bar([KEYS[i] for i in order], [averages[i] for i in order], align='center', color=[ 'C4', 'C2', 'C0', 'C3'])
    plt.ylim(0., 1.)
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35, rotation=30)
    plt.tight_layout()
    plt.savefig('sparse_total.png', dpi=300)


    # envs = ['densecabinet', 'denseantpush', 'denseantfall']
    # fig, axs = plt.subplots(1, len(envs), figsize=(6 * len(envs), 6))
    # if len(envs) == 1:
    #     axs = [axs]
    # id = ord('A')
    # for ax, env_name in zip(axs, envs):
    #     plot_env(ax, env_name, chr(id))
    #     id += 1
    # plt.tight_layout()
    # plt.savefig('dense.png', dpi=300)