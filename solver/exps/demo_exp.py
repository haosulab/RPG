from tools.sem import exp, run

exps = dict(
    decay_std=exp("python3 decay_std_demo.py -o"),

    twomode = exp("python3 train_line_explore.py -o --path twomode_demo"),
)
exps['twomode2'] = exps['twomode'].add('path twomode_demo2')
exps['twomode3'] = exps['twomode'].add('path twomode_demo3')

# learning std is not enough .. due to exploration issue/noise?
# TODO: maybe there is some way to make it success??
exps['learn_std'] = exps['decay_std'].add('head_cfg.std_scale 0.9').add('head_cfg.std_mode fix_learnable').add('std_decay', '0.')
exps['std_0.2'] = exps['decay_std'].add('head_cfg.std_scale 0.2').add('std_decay', '0.')

run('seed', add_path=False, **exps)