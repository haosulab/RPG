import wandb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
api = wandb.Api()

import matplotlib
font = {'size'   : 18}
matplotlib.rc('font', **font)


def download(runs, k='rewards'):
    out = {}
    for v in runs:
        run = api.run(f"/feyat/mujoco_rl/{v}")
        history = run.scan_history()
        index = []
        value = []
        for i in history:
            if k in i.keys():
                index.append(i['_step'])
                value.append(i[k])
        out[v] = pd.Series(value, index=index)
    return pd.DataFrame(out)

ant = {
    'mbrpg': ['2smiu1tw', 'ge1n35f6', '3jbajhm5', '3j73jw7y'],
    'mbrl': ['1zpxw6wa', '2zxo1hic', '22y82njn'],
}


cabinet = {
    'mbrpg': ['1346b4l5', '2c2297lm', 'gow1d1xw', '2jvi14zd'],
    'mbrl': ['2pt061cp', '26si2hkz', '195jkh2m'],
}

cheetah = {
    'mbrpg': ['2kzegq81', 'rc61ho7t', '29tqzokp'],#
}


humanoid = {
    'mbrpg': ['111lkbeu', '3hsj6wsy'],#
}

def draw_compare(inp, ax: plt.Axes, title: str, key='rewards', max_len=1e10, smooth=False, sac=None):
    outs = {}
    max_x = max_len
    for k, v in inp.items():
        if isinstance(v, list):
            v = download(v, key)
        v.index = v.index
        v = v.dropna()
        outs[k] = v
        max_x = min(max_x, v.index.max())

    for k, v in outs.items():
        v = v.loc[:max_x]
        if smooth:
            v = v.rolling(10).mean()
        mean = v.mean(axis=1).values
        std = v.std(axis=1).values
        outs[k] = {'mean': mean, 'std': std, 'x': np.arange(len(mean))}
        x = v.index / 1000

        if k == 'mbrl':
            kwargs = dict(color='tab:red')
        else:
            kwargs = {}
        ax.plot(x, mean, label=k, **kwargs)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    if sac is not None:
        ax.plot([0, v.index[-1]/1000], [sac, sac], label='sac@3M')
        

    
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_xlabel("simulator steps (x M)"); 
    ax.set_ylabel("avg traj return")



if __name__ == '__main__':
    fig, ax = plt.subplots(1, 4, figsize=(20 * 4, 20))
    draw_compare(cheetah, ax[0], '(a) Cheetah-v3', key='rewards', smooth=True, sac=15100, max_len=1000)
    draw_compare(humanoid, ax[1], '(b) Humanoid-v3', key='rewards', smooth=True, sac=6250, max_len=460)
    # plt.tight_layout()
    # plt.savefig('x.png')

    # fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5))
    draw_compare(ant, ax[2], '(c) AntMove')
    draw_compare(cabinet, ax[3], '(d) Cabinet')
    plt.tight_layout()
    plt.savefig('y.png')