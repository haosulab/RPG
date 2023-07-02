
from tools.sem import exp, run

exps = dict(
    sac=exp("python3 train_sac.py -o"),
    ppo=exp("python3 train_ppo.py -o"),
)

exps['ppo_sparse'] = exps['ppo'].add('env_name', 'FasterGripper')
exps['sac_sparse'] = exps['sac'].add('env_name', 'FasterGripper')

exps['sac_fixmove'] = exps['sac'].add('env_name', 'FixMove')
exps['ppo_fixmove'] = exps['ppo'].add('env_name', 'FixMove')


run('ids', add_path=True, **exps)