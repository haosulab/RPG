from tools.sem import exp, run

exps = dict(
    umaze = exp("python3 train_diffrl.py -o"),
)

exps['umaze_value_old'] = exps['umaze'].add('agent.value_optim.training_iter 100').add('agent.policy_optim.compute_value 1.')
exps['umaze_value_no_clip'] = exps['umaze_value_old'].add('agent.policy_optim.max_grad_norm 10000000.')
exps['umaze_value_append'] = exps['umaze_value_old'].add('agent.policy_optim.update_traj_buffer True')

exps['umaze_value_long'] = exps['umaze_value_old'].add('agent.policy_optim.training_iter 20').add('agent.num_trajs_per_epoch 30')



exps['gripper'] = exps['umaze'].add('env.TYPE GripperUmaze')

exps['ppo_gripper'] = exp("python3 train_ppo.py -o")
exps['ppo_umaze'] = exp("python3 train_ppo.py -o --env_name UMaze")

exps['sac_gripper'] = exp("python3 train_sac.py -o")

exps['ppo_gripper_expl'] = exps['ppo_gripper'].add('actor.head.std_scale 0.1')

exps['gd_umaze'] = exps['umaze']
exps['value_umaze'] = exps['umaze_value_old'].add('agent.num_trajs_per_epoch 10')

exps['value_umaze_0.99'] = exps['value_umaze'].add('agent.gamma 0.99')
exps['value_umaze_0.97'] = exps['value_umaze'].add('agent.gamma 0.97')
exps['value_umaze_less_expl'] = exps['value_umaze_0.97'].add('agent.head_cfg.std_scale 0.01')

exps['gd_gripper_less_expl'] = exps['gripper'].add('agent.head_cfg.std_scale 0.01')

for i in ['gd_umaze', 'value_umaze']:
    exps[i.replace('umaze', 'gripper')] = exps[i].add('env.TYPE GripperUmaze')

exps['value_sparse_gripper'] = exps['value_gripper'].add("env.TYPE SparseGripper")
exps['value_sparse_v2'] = exps['value_gripper'].add("env.TYPE SimpleStartGripper")

exps['value_on_policy_sparse'] = exps['value_sparse_v2'].add("agent.num_trajs_per_epoch 30").add('agent.value_optim.training_iter 40').add('agent.policy_optim.training_iter 10').add("agent.policy_optim.maxlen 300")

exps['value_on_u_gripper'] = exps['value_sparse_gripper'].add("agent.num_trajs_per_epoch 30").add('agent.value_optim.training_iter 40').add('agent.policy_optim.training_iter 10').add("agent.policy_optim.maxlen 300")




exps['value_on_lr_0.002'] = exps['value_on_u_gripper'].add('agent.policy_optim.lr 0.002')
exps['value_on_lr_0.0001'] = exps['value_on_u_gripper'].add('agent.policy_optim.lr 0.0001')

exps['value_on_expl_0.3'] = exps['value_on_u_gripper'].add('agent.head_cfg.std_scale 0.3')
exps['value_on_faster'] = exps['value_on_u_gripper'].add('env.TYPE FasterGripper')

exps['value_on_mlp_3'] = exps['value_on_faster'].add("agent.backbone.mlp_layers", '3').add('agent.policy_optim.lr 0.001')
exps['value_on_mlp_3_expl'] = exps['value_on_mlp_3'].add("agent.head_cfg.std_scale 0.3")

exps['value_on_15'] = exps['value_on_mlp_3'].add('agent.policy_optim.batch_size 15')
exps['value_on_15_clamp'] = exps['value_on_15'].add('agent.head_cfg.use_tanh False')

exps['value_on_15_0.0001'] = exps['value_on_15'].add('agent.policy_optim.lr 0.0001')
exps['value_on_15_penalty'] = exps['value_on_15'].add('agent.policy_optim.action_penalty 1.')
exps['value_on_15_penalty_2'] = exps['value_on_15'].add('agent.policy_optim.action_penalty 0.5')

exps['value_on_15_penalty_0001'] = exps['value_on_15_0.0001'].add('agent.policy_optim.action_penalty 0.5')

exps['value_on_15_penalty_0001_v2'] = exps['value_on_15_penalty_0001']
exps['value_on_15_penalty_0001_v3'] = exps['value_on_15_penalty_0001']

exps['value_on_15_penalty_00005_v1'] = exps['value_on_15_penalty_0001'].add('agent.policy_optim.lr 0.00005')
exps['value_on_15_penalty_00005_v2'] = exps['value_on_15_penalty_0001'].add('agent.policy_optim.lr 0.00005')
exps['value_on_15_penalty_00005_v3'] = exps['value_on_15_penalty_0001'].add('agent.policy_optim.lr 0.00005')
exps['value_on_15_less_penalty_00005_v1'] = exps['value_on_15_penalty_0001'].add('agent.policy_optim.lr 0.00005').add('agent.policy_optim.action_penalty 0.1')
exps['value_on_15_less_penalty_00005_v2'] = exps['value_on_15_penalty_0001'].add('agent.policy_optim.lr 0.00005').add('agent.policy_optim.action_penalty 0.1')

exps['value_on_15_penalty_0001_less'] = exps['value_on_15_0.0001'].add('agent.head_cfg.std_scale 0.05')
exps['value_on_15_penalty_0001_expl'] = exps['value_on_15_penalty_0001'].add('agent.head_cfg.std_scale 0.2')
exps['value_on_15_penalty_0003_expl'] = exps['value_on_15_penalty_0001_expl'].add('agent.policy_optim.lr 0.0003')

exps['value_on_15_penalty_0003'] = exps['value_on_15_penalty_0001'].add('agent.policy_optim.lr 0.0003')

exps['value_on_policy_sparse_2'] = exps['value_sparse_v2'].add("agent.num_trajs_per_epoch 10").add('agent.value_optim.training_iter 10').add('agent.policy_optim.training_iter 10').add("agent.policy_optim.maxlen 300").add("agent.value_optim.maxlen 1000").add("agent.backbone.mlp_layers", '3')


exps['gd_fixmove'] = exps['value_umaze'].add('agent.value_optim.training_iter 100').add('agent.policy_optim.compute_value 0.').add('agent.policy_optim.training_iter 10').add("agent.num_trajs_per_epoch 5").add("agent.policy_optim.batch_size 5").add('env.TYPE FixMove').add('agent.render_epoch 5').add('agent.policy_optim.action_penalty 0.5').add('agent.head_cfg.std_scale 0.01').add('agent.policy_optim.lr 0.0001')

exps['value_fixmove'] = exps['value_umaze'].add('agent.value_optim.training_iter 100').add('agent.policy_optim.compute_value 1.').add('agent.policy_optim.training_iter 10').add("agent.num_trajs_per_epoch 5").add("agent.policy_optim.batch_size 5").add('env.TYPE FixMove').add('agent.render_epoch 5').add('agent.policy_optim.action_penalty 0.5').add('agent.head_cfg.std_scale 0.01').add('agent.policy_optim.lr 0.0001')

#exps['test'] = exps['value_on_15_penalty_00005_v1']
exps['test'] = exps['value_on_15_penalty_0001']

exps['value_pusher'] = exps['value_on_15_penalty_00005_v1'].add('env.TYPE HardPush')


exps['pg_gripper_umaze'] = exps['value_on_15_penalty_00005_v1'].add('agent.policy_optim.policy_gradient 1.')
exps['pure_pg_gripper_umaze'] = exps['value_on_15_penalty_00005_v1'].add('agent.policy_optim.policy_gradient 1.').add('agent.policy_optim.gd 0.')

exps['pure_pg_gripper_umaze_v2'] = exps['pure_pg_gripper_umaze'].add('agent.policy_optim.lr 0.001')


exps['rnd_pusher'] = exps['value_pusher'].add('agent.use_rnd 1.')

run('ids', add_path=True, **exps)