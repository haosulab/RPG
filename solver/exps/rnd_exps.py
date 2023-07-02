from tools.sem import exp, run

exps = dict(
    base=exp('python3 train_rnd.py -o --agent.policy_optim.lr 0.001')
)

exps['gd'] = exps['base']
exps['pg'] = exps['base'].add('agent.policy_optim.policy_gradient', 1.).add('agent.policy_optim.gd 0.')

for std in [0.4, 0.6, 0.8]:
    for method in ['gd', 'pg']:
        exps[method + f'_{std}'] = exps[method].add('agent.head_cfg.std_scale', std)

run('seed', **exps, add_path=True)