import os
import glob
import copy
import pandas as pd
import numpy as np
from tools.config import Configurable, CN, merge_a_into_b, extract_variant
import matplotlib.pyplot as plt


sac_head = dict(
    head=dict(std_scale=1., std_mode='statewise', squash=True),
    pi_a=dict(ent=dict(coef=1., target_mode='auto'))
)

trainer_weights = dict(trainer=dict(weights=dict(reward=10., q_value=1., state=10000.,)), reward_scale=5., actor_delay=2) # sparse reward

base_config = dict(
    max_epoch=2000, # 200 * 5 * 2000
    steps_per_epoch=200,
    env_name='SmallMaze',
    env_cfg=dict(n=5, ignore_truncated_done=True),
    reward_scale=0.,
    buffer=dict(max_episode_num=100000),
    #z_dim=6,
    #z_cont_dim=0,
    hidden=dict(n=6),

    model=dict(qmode='value'),
    update_train_step=1,
    horizon=3,
    actor_delay=4, #10,
    z_delay=0,

    trainer=dict(weights=dict(reward=10000., q_value=100., state=1000.)),
    pi_a=dict(ent=dict(coef=0.0, target_mode='fixed'),),
    head=dict(
        linear=False,
        squash=True,
        std_mode='fix_no_grad',
        std_scale = 0.2
    ),
    pi_z=dict(ent=dict(coef=1000., target_mode='none'),),

    path='tmp/maze',
    hooks=dict(
        save_traj=dict(n_epoch=4, save_gif_epochs=10, occupancy=1), 
        # save_train_occupancy=dict(n_epoch=10)
    ),
    info=dict(coef=0.0),
    # info=dict(mutual_info_weight=0.03, action_weight=1., obs_weight=1., epsilon=0.01, std_mode='fix_no_grad'),

    _variants=dict(
        #sac=dict(model=dict(qmode='Q'), horizon=1, trainer=dict(weights=dict(state=0.))),
        value=dict(model=dict(qmode='value'), horizon=1),
        sac3=dict(model=dict(qmode='Q'), horizon=3),
        value3=dict(model=dict(qmode='value'), horizon=3, hidden=dict(n=1), head=dict(std_scale=0.2)),
        value3_2=dict(model=dict(qmode='value'), horizon=3, hidden=dict(n=1), head=dict(std_scale=0.1, squash=False)),
        #z=dict(model=dict(qmode='value'), horizon=3, z_dim=6, info=dict(coef=1.), pi_a=dict(pi=dict(head=dict(std_scale=0.1)))),
        z = dict(_inherit='value3', hidden=dict(n=6), info=dict(coef=0.1)),
        z2=dict(_inherit='z', head=dict(std_scale=0.05)),
        relabelz=dict(_inherit='z2', relabel=0.8),

        rnd = dict(
            _inherit='z2',
            rnd=dict(scale=1.),
        ),
        rnd2=dict(_inherit='rnd', head=dict(std_scale=0.3, std_mode='statewise')),
        # medium=dict(_inherit='rnd2', env_name='MediumMaze'),
        medium2=dict(_inherit='rnd2', env_name='MediumMaze', head=dict(std_scale=0.2, std_mode='fix_no_grad', linear=False, squash=False), pi_a=dict(ent=dict(coef=0.01)), rnd=dict(scale=1.), info=dict(coef=0.03), path='tmp/medium'), # seems that we can continue to decrease the info coef
        medium0=dict(_inherit='medium2', z_dim=1, path='tmp/medium0'),
        lessinfo=dict(_inherit='medium2', info=dict(coef=0.03), path='tmp/lessinfo'),

        rndreward=dict(_inherit='medium2', rnd=dict(
            as_reward=True,
            training_on_rollout=False,
            obs_mode='obs',
            scale=0.1,
        ),
        env_cfg=dict(obs_dim=8),
        path='tmp/rndreward'),

        small=dict(_inherit='rndreward', env_cfg=dict(n=5), path='tmp/small2'),
        rndz=dict(_inherit='small', rnd=dict(include_latent=True), path='tmp/rndz'),
        rndz_01=dict(_inherit='rndz', info=dict(coef=0.1), path='tmp/rndz_01'),
        small_optimz=dict(_inherit='small', 
            z_delay=8,
            pi_z=dict(ent=dict(coef=5., target_mode='none', schedule=dict(TYPE='linear', min_value=1., end=1000))), path='tmp/small_optimz'),
        #small_optimz2=dict(_inherit='small_optimz', pi_z=dict(ent=dict(coef=0.1)), path='tmp/small_optimz2'),; 0.1 is too small.
        small_optimz2=dict(_inherit='small_optimz', pi_z=dict(ent=dict(coef=2.)), path='tmp/small_optimz2'),

        ant=dict(
            _inherit='small', env_cfg=dict(n=1, obs_dim=8), info=dict(coef=0.01), env_name='AntMaze', path='tmp/antmaze', save_video=1000,
            head=dict(linear=False, squash=False),
            hooks=dict(
                save_traj=dict(n_epoch=40, save_gif_epochs=10, occupancy=1), 
                # save_train_occupancy=dict(n_epoch=10)
            ),
            max_epoch=200000,
            steps_per_epoch=150,
            buffer=dict(max_episode_num=10000),
            pi_a=dict(ent=dict(coef=0.01 * 0.001))
        ),
        ant_squash=dict(
            _inherit='ant', head=dict(squash=True, linear=False, std_scale=0.4), path='tmp/antmaze_squash',
        ),
        ant_nornd=dict(_inherit='ant_squash', rnd=dict(scale=0.), info=dict(coef=10.), path='tmp/antmaze_nornd'),
        ant_nornd2=dict(_inherit='ant_nornd',  path='tmp/antmaze_nornd2', hidden=dict(use_next_state=True, action_weight=0.)),
        ant_nornd3=dict(_inherit='ant_nornd2',  path='tmp/antmaze_nornd3',  head=dict(std_scale=0.3, squash=False)),


        ant_nornd_zdim20=dict(_inherit='ant_nornd', hidden=dict(n=20), path='tmp/antmaze_zdim20'),
        ant_nornd_normal = dict(_inherit='ant_nornd', path='tmp/antmaze_nornd_normal', trainer=dict(weights=dict(q_value=1.)), hidden=dict(TYPE='Gaussian', n=5)),

        ant_state = dict(_inherit='ant_nornd_normal', path='tmp/ant_state', info=dict(coef=1.), hidden=dict(use_next_state=True, action_weight=0.)),
        ant_explore = dict(_inherit='ant_state', path='tmp/ant_explore', head=dict(std_scale=1.2)),

        ant_maxent = dict(_inherit='ant_explore', path='tmp/ant_maxent', head=dict(std_scale=1.0, std_mode='statewise'), pi_a=dict(ent=dict(coef=1., target_mode='none'))),

        gaussian=dict(_inherit='small', env_cfg=dict(n=1),
            hidden=dict(TYPE='Gaussian', n=5), 
            info=dict(coef=0.001, weight=dict(TYPE='linear', min_value=1., end=8000)), path = 'tmp/gaussian_maze',
        ),

        ant_gaussian=dict(
            _inherit='ant_squash',
            hidden=dict(TYPE='Gaussian', n=5), 
            info=dict(coef=0.001), path = 'tmp/gaussian_maze',
        ),

        antcross = dict(_inherit='ant_squash', env_name='AntMaze3', path='tmp/antcross', steps_per_epoch=200),
        block=dict(_inherit='ant_squash', env_cfg=dict(n=1, obs_dim=5), env_name='BlockPush', path='tmp/block'),

        arm=dict(steps_per_epoch=None, _inherit='ant_squash', reward_scale=0., env_cfg=dict(n=1, obs_dim=6), env_name='FixArm', path='tmp/arm'),
        eearm=dict(_inherit='arm', env_name='EEArm', path='tmp/eearm'),
        eearm_gaussian=dict(_inherit='eearm', hidden=dict(TYPE='Gaussian', n=12), env_cfg=dict(n=5), info=dict(coef=0.0008), path='tmp/eearm_guassian'),
        eearm_rew=dict(_inherit='eearm', reward_scale=1., rnd=dict(scale=0.0), hidden=dict(n=1), env_cfg=dict(reward_type='dense'), path='tmp/eearm_reward'),

        arm_rew = dict(_inherit='arm', reward_scale=1., rnd=dict(scale=0.0), hidden=dict(n=1), env_cfg=dict(reward_type='dense',), path='tmp/arm_reward'),
        arm_sparse = dict(_inherit='arm', reward_scale=0.01, rnd=dict(scale=0.1), hidden=dict(n=6), env_cfg=dict(reward_type='sparse'), path='tmp/arm_sparse'),
        cabinet=dict(_inherit='ant_squash', env_cfg=dict(n=1, obs_dim=5), env_name='Cabinet', path='tmp/cabinet'),

        
        # method base, to test on different algorithms
        mbsac=dict(
            _inherit='eearm_rew', rnd=dict(scale=0.), info=dict(coef=0.), 
            env_cfg=dict(obs_dim=6),
            hidden=dict(n=1, TYPE='Categorical'), path=None,
            **sac_head,
            **trainer_weights,
        ),

        mbsacrnd=dict(_inherit='mbsac', rnd=dict(scale=0.1)),
        mbddpgrnd=dict(_inherit='eearm_rew', rnd=dict(scale=0.1), info=dict(coef=0.), hidden=dict(n=1, TYPE='Categorical'), 
                       path=None, **trainer_weights),
        mbsacv2rnd=dict(_inherit='mbsacrnd', trainer=dict(discard_ent=True)),
        mbsaclowstd=dict(_inherit='mbsacrnd', pi_a=dict(ent=dict(coef=0.01)), head=dict(std_scale=0.4)),
        mbsacv3=dict(_inherit='mbsaclowstd', trainer=dict(discard_ent=True)),
        #mbsacrnd5=dict(_inherit='mbsacrnd', env_cfg=dict(n=5)),

        rpgc=dict(_inherit='eearm_gaussian', **trainer_weights, rnd=dict(scale=0.1), hidden=dict(n=12), info=dict(coef=0.001), path=None, env_cfg=dict(n=1)),
        rpgd=dict(_inherit='rpgnormal', hidden=dict(TYPE='Categorical'), path=None),
        rpgcv2=dict(_inherit='mbsacv3', hidden=dict(n=12, TYPE='Gaussian'), info=dict(coef=0.001)),

        rpgcv3=dict(_inherit='rpgcv2', buffer=dict(max_episode_num=500)),
        rpgdelay = dict(_inherit='rpgcv2', info_delay=1),

        rpgdv2=dict(_inherit='rpgcv2', hidden=dict(TYPE='Categorical')),

        rpgdv3=dict(
            _inherit='rpgdv2', 
            pi_z=dict(ent=dict(coef=1., target_mode='none'),)
        ),
        mpc=dict(_inherit='mbsacv3', cem=dict()),

        rewardrpg=dict(_inherit='rpgdv3', use_reward_schedule=True),
        rewardrpgc=dict(_inherit='rpgcv2', use_reward_schedule=True),

        rewardsac=dict(_inherit='mbsacv3', use_reward_schedule=True),
        # rpgnormal1=dict(_inherit='rpgnormal', env_cfg=dict(n=1)),
        # rpgdiscrete1=dict(_inherit='rpgdiscrete', env_cfg=dict(n=1)),
        # rpgsac_discard=dict(_inherit='rpgsac', trainer=dict(discard_ent=True)),
        rpgoptim=dict(
            _inherit='rpgdv3',
            pi_z=dict(ent=dict(coef=0.01),),
            z_delay=2
        ),

        rpgmix=dict(_inherit='rpgcv2', hidden=dict(TYPE='Mixture', n=12)),
        sacgmm=dict(_inherit='mbsacv3', head=dict(TYPE='GMM')),
        sacflow=dict(_inherit='mbsacv3', head=dict(TYPE='Flow')),

        relabel=dict(_inherit='rpgcv2', relabel_latent='exp(0.,0.99,10000)',
                     hidden=dict(action_weight=0.,  use_next_state=True, head=dict(std_mode='statewise', maximal_std_val=1., minimal_std_val=0.05))),
        relabeld = dict(_inherit='rpgcv2', relabel_latent='linear(0.,0.99,1000)',
                     hidden=dict(action_weight=0., use_next_state=True, head=dict(epsilon=0.01), TYPE='Categorical')),
    ),
)


class Experiment(Configurable):
    def __init__(self, base_config, cfg=None, path='maze_exp', wandb=False, opt=None) -> None:
        super().__init__()
        self.basis = base_config.pop('_variants')

        from rpg.soft_rpg import Trainer
        cfg = Trainer.dc
        merge_a_into_b(CN(base_config), cfg)
        self.base_config = cfg
        self.env_configs = self.get_env_configs()
        self.wandb = wandb
        self.path = path
        self.exps = {}
        self.opt = opt

    def add_exps(self, expname, cfgs, names=None, base=None, default_env=None):
        if names is not None:
            cfgs['_names'] = names
        self.exps[expname] = {
            'cfgs': cfgs,
            'base': base,
            'default_env': default_env,
        }

    def get_variants(self):
        return self.basis

    def get_env_configs(self):
        # name_name, env_config
        return {
            'SmallMaze': dict(n=5),
            'MediumMaze': dict(n=5),
            'TreeMaze': dict(n=5),
            'AntMaze': dict(n=1, obs_dim=8),
        }

    def build_configs(self, env_name, expname, verbose=False):
        """
        base
        To specify the variants, also use dictionary, but for certain keys it can be a list. We only support list of the same size.
            - {
                **config
                _default: None
                _names: [] a list of names
              }
              then zip them together
        """
        exp_config = copy.deepcopy(self.exps[expname])
        base = exp_config.pop('base', None)
        cfgs = exp_config.pop('cfgs')
        if env_name is None:
            env_name = exp_config.pop('default_env', None)
        #assert env_name is not None


        _variants = cfgs.pop('_variants', None)

        if len(cfgs) > 0:
            names = cfgs.pop('_names', [])
            names = [[i] for i in names]
            rename = len(names) == 0
                
            factor_name = []
            default = {}

            variants = []
            def set_keyval(d, keys, val):
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = val

            def process_variants(configs, keys):
                for k, v in configs.items():
                    new_keys = keys + k.split('.')
                    if isinstance(v, list):
                        factor_name.append(k)
                        if len(variants) == 0:
                            for i in range(len(v)):
                                variants.append(copy.deepcopy(default))
                        if not rename:
                            assert len(variants) == len(names), "The length of the list should be the same."
                        else:
                            if len(names) == 0:
                                names.extend([[] for i in range(len(v))])
                            for i in range(len(v)):
                                names[i].append(str(v[i]))
                        assert len(variants) == len(v), "The length of the list should be the same."
                        for i in range(len(v)):
                            set_keyval(variants[i], new_keys, v[i])
                    elif isinstance(v, dict):
                        process_variants(v, new_keys)
                    else:
                        if len(variants) == 0:
                            set_keyval(default, new_keys, v)
                        else:
                            for i in range(len(variants)):
                                set_keyval(variants[i], new_keys, v)
                                
            process_variants(cfgs, [])


            factor_name = '_'.join(factor_name)
            if rename:
                names = [factor_name + '_'.join(n) for n in names]
            else:
                names = [n[0] for n in names ]
            print(len(variants), len(names))
        else:
            assert _variants is not None
            assert base is None

            _base = _variants.pop('_base', None)
            names = []
            variants = []
            for k, v in _variants.items():
                v = CN(v)

                # priority: v > _base > v._base
                if _base is not None:
                    var = CN(copy.deepcopy(_base))
                    var.set_new_allowed(True)
                    merge_a_into_b(v, var)
                    v = var

                if '_base' in v:
                    var = extract_variant(v.pop('_base'), self.get_variants())
                    var.set_new_allowed(True)
                    merge_a_into_b(v, var)
                    v = var

                names.append(k)
                variants.append(v)

        if verbose:
            print("name", names)
            for i in range(len(variants)):
                print(variants[i])


        cfg = self.base_config.clone()
        cfg.defrost()

        if base is not None:
            var = extract_variant(base, self.get_variants())
            cfg.set_new_allowed(True)
            merge_a_into_b(var, cfg)
            

        outputs = []

        env_cfg = None
        if 'env_name' in self.env_configs:
            
            env_cfg = self.env_configs[env_name]

        for name, k in zip(names, variants):
            if '_base' in k:
                var_cfg = self.base_config.clone()
                var_cfg.defrost()
                var = extract_variant(k.pop('_base'), self.get_variants())
                var_cfg.set_new_allowed(True)
                merge_a_into_b(var, var_cfg)
            else:
                var_cfg = cfg.clone()
            var_cfg.set_new_allowed(True)

            k = CN(k)
            kws = dict(env_name=env_name)
            if env_cfg is not None:
                kws['env_cfg'] = env_cfg
            merge_a_into_b(
                CN(kws), var_cfg
            )
            merge_a_into_b(k, var_cfg)
            var_cfg.env_name = env_name
            cfg_name = f'{env_name}_{expname}_{name}'

            var_cfg.set_new_allowed(True)
            if self.wandb:
                var_cfg.wandb = CN({'project': self.path, 'name': cfg_name, 'group': cfg_name})
            else:
                var_cfg.path = os.path.join(self.path, expname, cfg_name)
                if 'MODEL_PATH' in os.environ:
                    var_cfg.path = os.path.join(os.environ['MODEL_PATH'], var_cfg.path)
                var_cfg.log_date = True

            if self.opt is not None:
                merge_a_into_b(self.opt, var_cfg)

            outputs.append(var_cfg)

        return outputs
        

    def add_var(self, k, d):
        assert k not in self.base_config['_variants']
        self.base_config['_variants'][k] = d

    def run_config(self, cfg):
        from rpg.soft_rpg import Trainer
        trainer = Trainer(None, cfg=cfg)
        trainer.run_rpgm()

    def plot(self, configs, keyword):
        outputs = {}
        plt.clf()
        for i in configs:
            files = glob.glob(os.path.join(i.path, "*"))
            if len(files) > 0:
                filename = sorted(files, key=os.path.getmtime)[-1]

                frame = pd.read_csv(os.path.join(filename, 'progress.csv'))
                progress = frame[keyword].dropna()

                key = i.path.split('/')[-1].split('_')[2]
                outputs[key] = progress.to_numpy()
                plt.plot(np.arange(len(progress)), progress, label=key)

        plt.legend()
        plt.savefig('x.png')

        
    def main(exp):
        args, _unkown = exp.parser.parse_known_args()

        exps = args.exp.split(',')

        configs = []
        for i in exps:
            configs.append(exp.build_configs(args.env_name, i, verbose=True)) # inherit from small
        # with open("config.txt", "w") as f:
        #     f.write(str(configs[0][0]))
        # breakpoint()
            
        if args.runall is not None:
            # run on cluster
            import sys
            for expname, configs in zip(exps, configs):
                def make_base(configs, seed=None):
                    env_name = configs.env_name
                    arguments = [sys.argv[0], '--env_name', env_name, '--exp', expname, '--wandb', str(exp._cfg.wandb)]
                    base = ' '.join(arguments)
                    if seed is not None:
                        base = base + ' --seed ' + seed
                    return base

                if args.seed is None or args.runall == 'remote_parallel':
                    seeds = [None]
                else:
                    split = ',' if ',' in args.seed else '_'
                    seeds = args.seed.split(split)

                ids = list(range(len(configs)))
                if args.ids is not None:
                    split = ','
                    ids = [int(i) for i in args.ids.split(split)]
                        
                    
                workers = []
                for seed in seeds:

                    if args.runall == 'local':
                        for i in range(len(configs)):
                            if i not in ids:
                                continue
                            cmd = 'python3 '+ make_base(configs[i], seed) + ' --id '+str(i)
                            # print('running ', cmd, 'yes/no?')
                            # x = input()
                            # if x == 'yes':
                            # os.system(cmd)
                            print(cmd)
                    elif args.runall == 'parallel': # run in local in parallel ..
                        def work(cmd):
                            os.system(cmd)

                        assert args.id is not None
                        #ids = [args.id]
                        #for i in range(len(configs)):
                        i =  args.id
                        base = make_base(configs[i], seed)
                        seed_info = '-seed-' + str(seed) if seed is not None else ''
                        cmd = 'python3 '
                        cmd =  cmd + base + ' --id '+str(i) + ' --job_name {}-{}{} '.format(expname, i, seed_info)

                        from multiprocessing import Process
                        workers.append(Process(target=work, args=(cmd,)))
                        workers[-1].start()

                    else:
                        assert args.runall == 'remote_parallel' or args.runall == 'remote'
                        for i in range(len(configs)):
                            if i not in ids:
                                continue
                            base = make_base(configs[i], seed)
                            silent = ' --silent ' if args.silent else ''
                            seed_info = '-seed-' + str(seed) if seed is not None else ''
                            cmd = 'remote.py --go '
                            cmd = cmd + ' --cpu ' + str(args.cpu) + ' '
                            cmd =  cmd + silent +base + ' --id '+str(i)
                            if args.runall == 'remote_parallel':
                                seed_info = str(args.seed).replace(',', '_')
                                cmd = cmd + ' --runall parallel --seed  ' + str(args.seed).replace(',', '_')
                            cmd = cmd +  ' --job_name {}-{}{} '.format(expname, i, seed_info.replace('_','x'))
                            os.system(cmd)

                for w in workers:
                    w.join()
                    
        else:
            configs = sum(configs, [])
            if args.id is not None or args.ids is not None:
                exp_id = args.id
                if args.ids is not None:
                    assert args.id is None
                    for idx, i in enumerate(configs):
                        if i['path'].split('_')[-1] == args.ids:
                            exp_id = idx

                if args.download:
                    os.system('kubectl cp hza-try:/cephfs/hza/models/{} {}'.format(configs[exp_id].path, configs[exp_id].path))
                    exit(0)

                cfg = configs[exp_id]
                cfg.seed = args.seed
                if args.seed is not None and cfg.path is not None:
                    cfg.path += '_seed' + str(args.seed)

                exp.run_config(cfg)
            else:
                for idx, i in enumerate(configs):
                    print(idx, i.path)
                if args.download:
                    for i in range(len(configs)):
                        os.system('kubectl cp hza-try:/cephfs/hza/models/{} {}'.format(configs[i].path, configs[i].path))
                    exit(0)
                exp.plot(configs, 'success' or args.key)


def build_exp(base_config, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', default=None, type=str, help='env name')
    parser.add_argument('--exp', default=kwargs.get('exp', 'zdim'), type=str, help='experiment to run')

    parser.add_argument('--id', default=0, type=int, help='id')
    parser.add_argument('--ids', default=None, type=str, help='ids')
    parser.add_argument('--runall', default=None, type=str)
    parser.add_argument('--download', action='store_true', help='download data')
    parser.add_argument('--silent', action='store_true', help='silent')
    parser.add_argument('--cpu', default=2, type=int)

    parser.add_argument('--seed', default=None)
    parser.add_argument('--key', default=None)

    exp = Experiment.parse(base_config, parser=parser)
    exp.parser = parser
    return exp


if __name__ == '__main__':
    exp = build_exp(base_config)
    exp.add_exps(
        'zdim', dict(hidden=dict(n=[1, 3, 6, 2])), ['rl', 'rpg1', 'rpg2', 'rpg3'], base='small',
    )

    # python3 maze_exp.py --exp buffers  --wandb True --runall nautilus
    exp.add_exps(
        'relabel', dict(relabel=[0.8, 0.1]), ['8', '1'], base='small',
    )
    exp.add_exps(
        'rndbuf', dict(rnd=dict(buffer_size=[10000, 100000, int(1e6)])), ['bs4', 'bs5', 'bs6'], base='small',
    )
    exp.add_exps(
        'stda', dict(head=dict(std_scale=[0.1, 0.2, 0.3])), base='small',
    )

    exp.add_exps(
        'infoloss', dict(info=dict(coef=[0.02, 0.05, 0.08, 0.1])), base='small',
    )

    exp.add_exps(
        'treerl', dict(hidden=dict(n=[1, 6]), info=dict(coef=0.02)), names=['rl', 'rpg'], base='small',
    )

    exp.add_exps(
        'treerl2', dict(relabel=[0., 0., 0.8], hidden=dict(n=[1, 6, 6]), env_cfg=dict(n=1), info=dict(coef=0.02)), names=['rl', 'rpg', 'relabel'], base='small',
    )

    exp.add_exps(
        'bufrl', dict(rnd=dict(buffer_size=[1000, 10000, 100000, int(1e6)]), hidden=dict(n=1)), ['bs3', 'bs4', 'bs5', 'bs6'], base='small',
    )

    exp.add_exps(
        'normalinfo', dict(info=dict(coef=[0.001, 0.005, 0.01, 0.05, 0.1])), base='gaussian',
    )

    exp.add_exps(
        'antsquashstd', dict(head=dict(std_scale=[0.2, 0.3, 0.4, 0.5])), base='ant_squash',
    )

    exp.add_exps(
        'antstd', dict(head=dict(std_scale=[0.2, 0.3, 0.4, 0.5])), base='ant',
    )

    exp.add_exps(
        'antinfo', dict(info=dict(coef=[0.01, 0.05, 0.1, 0.001])), base='ant',
    )

    exp.add_exps(
        'antrl', dict(hidden=dict(n=[1, 6, 6, 6, 6, 6]), info=dict(coef=[0., 0., 0.01, 0.05, 0.1, 0.001])),
                    names=['rl', 'rpg0.', 'rpg0.01', 'rpg0.05', 'rpg0.1', 'rpg0.001'], base='ant',
        default_env='AntMaze',
    )


    exp.add_exps(
        'ant2rl', dict(hidden=dict(n=[1, 6, 6]), info=dict(coef=[0., 0.05, 0.001])),
                    names=['rl', 'rpg0.05', 'rpg0.001'], base='ant_squash',
        default_env='AntMaze2',
    )

    exp.add_exps(
        'ant2rlv2', dict(hidden=dict(n=[1, 6, 6, 6]), info=dict(coef=[0., 0.05, 0.1, 0.5])),
                    names=['rl', 'rpg0.05', 'rpg0.1', 'rpg0.5'], base='ant_squash',
        default_env='AntMaze2',
    )

    exp.add_exps(
        'ant2nornd', dict(hidden=dict(n=[6, 6, 6, 6, 6, 6]), info=dict(coef=[0.05, 0.1, 0.5, 1., 5., 10.])), base='ant_nornd',
        default_env='AntMaze2',
    )

    exp.add_exps(
        'ant2normal', dict(info=dict(coef=[0.001, 0.005, 0.01,0.0001])), base='ant_gaussian', default_env='AntMaze2',
    )

    exp.main()