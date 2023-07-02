# run halfcheetah with SAC ..
#python3 maze_tester/test_maze.py --var mbsac --env_name HalfCheetah-v3 --env_cfg.n 1 --path tmp/mbsac_cheetah --hooks.save_traj.n_epoch 1000000000 --hooks.save_train_occupancy.n_epoch 10000000000 --trainer.weights.reward 1. --trainer.weights.q_value 1.
# python3 maze_tester/test_maze.py --var mbsac --env_name AdroitHammer --env_cfg.n 1 --env_cfg.reward_type dense --path tmp/mbsac_hammer_dense --hooks.save_traj.n_epoch 1000000000
python3 maze_tester/test_maze.py --var rpgnormal --env_name EEArm --env_cfg.reward_type sparse --path tmp/rpgnormal_cabinet
#python3 maze_tester/test_maze.py --var rpgnormal --env_name AdroitHammer --env_cfg.reward_type sparse --path tmp/hammer --reward_scale 5. --trainer.weights.q_value 1. --trainer.weights.reward 10.