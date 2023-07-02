import os
from tools.sem import exp, run
from solver import MODEL_PATH

exps = dict(
    base=exp('python3 flow_estimator.py -o --format_strs csv+tensorboard')
)

exps['gd'] = exps['base'].add("--task GD")
exps['pg'] = exps['base'].add('--task PG')
exps['pggd'] = exps['base'].add("--task PGGD")
#exps['pggd_res'] = exps['pggd'].add("--residual 1")

#for std in [0.4, 0.6, 0.8]:
#    for method in ['gd', 'pg']:
#        exps[method + f'_{std}'] = exps[method].add('agent.head_cfg.std_scale', std)
for batch_size in [1, 128]:
    for method in ['gd', 'pg', 'pggd']:
        exps[method + f'_{batch_size}'] = exps[method].add('batch_size', batch_size)

for i in exps:
    exps[i] = exps[i].add("path", f"{MODEL_PATH}/FLOWTEST/{i}")

run('seed', **exps, add_path=True, default_exps='gd,pg,pggd')