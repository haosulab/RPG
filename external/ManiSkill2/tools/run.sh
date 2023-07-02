# python tools/replay_trajectory3.py --traj-path demos/rigid_body_envs/OpenCabinetDoor-v1/1018/link_1/trajectory.h5 --save-video --vis
# python tools/replay_trajectory3.py --traj-path demos/rigid_body_envs/OpenCabinetDrawer-v1/1004/link_0/trajectory.h5 --save-video --vis
#python3 tools/reply.py --kuafu
#python3 tools/reply.py
python3 tools/replay.py --kuafu --env_id 'drawer'
python3 tools/replay.py --env_id 'drawer'