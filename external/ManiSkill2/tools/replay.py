import gym
import os
import torch
import tqdm
import numpy as np
from mani_skill2.utils.wrappers import RecordEpisode
from tools.utils import animate
from mani_skill2.utils.sapien_utils import get_entity_by_name, look_at
from copy import deepcopy
import multiprocessing as mp
import PIL.Image as im

mp.set_start_method("spawn")


def sample_unit_ball_north():
    xyz = np.random.randn(3)
    if xyz[2] < 0:
        xyz *= -1
    return xyz / np.linalg.norm(xyz)


def sample_unit_ball_north_():
    xyz = np.random.randn(3)
    if xyz[2] < 0:
        xyz[2] *= -1
    if xyz[0] > 0:
        xyz[0] *= -1
    return xyz / np.linalg.norm(xyz)


def env_render(env, pose=None):
    # scene_center = np.array([-0.2,0.,0.2])
    # radius = 1.0
    scene_center = np.array([0.,0.,0.3])
    # scene_center = np.array([-0.3,0.,0.3])
    scene_center = np.array([-0.6,0.,0.3])
    radius = 2.5
    # radius = 1.6
    # radius = 1.0
    # radius = 2.0
    # radius = 1.0


    # pose = look_at(radius*sample_unit_ball_north()+scene_center, scene_center)
    if pose is None:
        pose = look_at(radius*sample_unit_ball_north_()+scene_center, scene_center)
    env.render_camera.set_local_pose(pose)

    pose_new = pose.to_transformation_matrix()
    r = pose_new[:3,:3] @np.array([[0,-1,0],[0,0,1],[-1,0,0]]).T
    pose_new[:3,:3] = r
    pose_ori = deepcopy(env.render_camera.get_model_matrix())
    return pose, pose_new



def render_traj(env_id, enable_kuafu, env_kwargs, states, poses=None):
    env = gym.make(env_id, enable_kuafu=enable_kuafu, **env_kwargs)
    output = {'image': [], 'pose': [], 'depth': [], 'actor_seg': [], 'camera_pose': []} 
    env.reset()

    env_render(env)

    idx = 0
    for qpos, state in tqdm.tqdm(states, total=len(states)):
        env.step(env.action_space.sample())
        env.cabinet.set_qpos(qpos)
        env.agent.set_state(state)

        if poses is None:
            pose, camera_pose = env_render(env)
            output['pose'].append(pose)
            output['camera_pose'].append(camera_pose)
            output['image'].append(env.render('rgb_array'))
        else:
            env_render(env, poses[idx])

        if not enable_kuafu:
            img = env.render('rgb_array')
            env.update_render()
            env.render_camera.take_picture()
            ret = env.unwrapped._get_camera_images(
                env.render_camera, rgb=False, depth=True, visual_seg=True, actor_seg=True
            )
            output['depth'].append(ret['depth'])
            output['actor_seg'].append(ret['actor_seg'].astype(np.uint8)[..., 0])

        idx += 1
    output['intrinsic'] = env.render_camera.get_intrinsic_matrix()
    return output


def plot(traj, output_path):

    if 'pose' in traj and len(traj['pose']) > 0:
        animate(traj['image'], os.path.join(output_path, 'output.mp4'))
        for idx, img in enumerate(traj['image']):
            alpha = ((img == 170).sum(axis=2) != 3).astype(np.uint8) * 255
            img = np.concatenate((img, alpha[:,:,None]), axis=2)

            img_name = f"0_{idx:03d}.png"
            im.fromarray(img).save(os.path.join(output_path, img_name))

            torch.save(traj['pose'], os.path.join(output_path, 'pose.pth'))
            np.savetxt(os.path.join(output_path, f'0_{idx:03d}.txt'), traj['camera_pose'][idx])

        np.savetxt(os.path.join(output_path, f'0_intrinsic.txt'), traj['intrinsic'])
    else:
        for idx in range(len(traj['depth'])):
            img_name = f"0_{idx:03d}_depth.npy"
            np.save(os.path.join(str(output_path), img_name), traj['depth'][idx])

            img_name = f"0_{idx:03d}_seg.png"
            im.fromarray(traj['actor_seg'][idx]).save(os.path.join(str(output_path), img_name))


            img_name = f"0_{idx:03d}_seg_c.png"
            import matplotlib.pyplot as plt
            colors = np.array(plt.get_cmap('tab20').colors)
            colors = colors[traj['actor_seg'][idx] % 20]
            im.fromarray((colors * 255).astype(np.uint8)).save(os.path.join(str(output_path), img_name))
    
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, default='door')
parser.add_argument('--kuafu', action='store_true')
args = parser.parse_args()

if args.env_id == 'door':
    traj_path = 'demos/rigid_body_envs/OpenCabinetDoor-v1/1018/link_1/'
    env_id = 'OpenCabinetDoor-v1'
    pid = '1018'
else:
    traj_path = 'demos/rigid_body_envs/OpenCabinetDrawer-v1/1004/link_0/'
    pid = '1004'
    env_id = 'OpenCabinetDrawer-v1'

env_kwargs = {'reward_mode': 'sparse', 'obs_mode': 'state', 'model_ids': [pid], 'fixed_target_link_idx': 1}

old_state = torch.load(traj_path + 'states')
new_state = []

n_interpolate = 5

for i in range(len(old_state) - 1):
    q1, s1 = old_state[i]
    q2, s2 = old_state[i+1]

    for j in range(n_interpolate):
        q = q1 + (q2 - q1) * j / n_interpolate
        for k, v in s1.items():
            print(k, type(v))
        s = {k: v + (s2[k] - v) * j / n_interpolate for k, v in s1.items() if k != 'robot_root_pose' and k!= 'controller'}
        s['robot_root_pose'] = s1['robot_root_pose']
        s['controller'] = s1['controller']

        new_state.append([q, s])
new_state.append(old_state[-1])
state = new_state #[:10]


if args.kuafu:
    traj = render_traj(
        env_id, True, env_kwargs, state, 
    )
else:
    traj = render_traj(
        env_id, False, env_kwargs,  state, torch.load(traj_path + 'pose.pth')
    )
plot(traj, traj_path)