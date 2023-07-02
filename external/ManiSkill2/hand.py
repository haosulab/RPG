# pip install --upgrade PyMCubes
import sapien.core as sapien
import torch
import PIL.Image as im
import numpy as np

engine = sapien.Engine()  # Create a physical simulation engine


use_kuafu=False

if use_kuafu:
    render_config = sapien.KuafuConfig()
    render_config.spp = 256
    render_config.use_denoiser = False

    renderer = sapien.KuafuRenderer(render_config)
    sapien.KuafuRenderer.set_log_level('warning')
else:
    renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)


scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
scene.set_timestep(1 / 10000.0)  # Set the simulation frequency


def add_camera_light(light=False):
    #camera_mount = scene.create_actor_builder().build_kinematic()
    camera = scene.add_camera(
        name="camera",
        width=512,
        height=512,
        fovy=1,
        near=0.01,
        far=10,
    )

    # camera_mount.set_pose(
    #     Pose(np.array([-0.5, -0.5, 1.0]), [0.8876263, -0.135299, 0.3266407, 0.2951603]))

    if light:
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_directional_light([0, 0.5, -1], color=[3.0, 3.0, 3.0])
    return camera


# def add_ground():
#     ground_material = renderer.create_material()
#     ground_material.base_color = np.array([202, 164, 114, 256]) / 256
#     ground_material.specular = 0.5
#     scene.add_ground(0, render_material=ground_material)
#     scene.set_timestep(1 / 240)
def modify_robot_visual(robot: sapien.Articulation):
    robot_name = robot.get_name()
    if "mano" in robot_name:
        return robot
    arm_link_names = [f"link{i}" for i in range(1, 8)] + ["link_base"]
    for link in robot.get_links():
        if link.get_name() in arm_link_names:
            pass
        else:
            for geom in link.get_visual_bodies():
                for shape in geom.get_render_shapes():
                    mat_viz = shape.material
                    mat_viz.set_specular(0.07)
                    mat_viz.set_metallic(0.3)
                    mat_viz.set_roughness(0.2)
                    if 'adroit' in robot_name:
                        mat_viz.set_specular(0.02)
                        mat_viz.set_metallic(0.1)
                        mat_viz.set_base_color(np.power(np.array([0.9, 0.7, 0.5, 1]), 1.5))
                    elif 'allegro' in robot_name:
                        if "tip" not in link.get_name():
                            mat_viz.set_specular(0.8)
                            mat_viz.set_base_color(np.array([0.1, 0.1, 0.1, 1]))
                        else:
                            mat_viz.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
                    elif 'svh' in robot_name:
                        link_names = ["right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r", "right_hand_q",
                                      "right_hand_e1"]
                        if link.get_name() not in link_names:
                            mat_viz.set_specular(0.02)
                            mat_viz.set_metallic(0.1)
                    elif 'ar10' in robot_name:
                        if "tip" in link.get_name():
                            mat_viz.set_base_color(np.array([20, 20, 20, 255]) / 255)
                            mat_viz.set_metallic(0)
                            mat_viz.set_specular(0)
                            mat_viz.set_roughness(1)
                        else:
                            color = np.array([186, 54, 56, 255]) / 255
                            mat_viz.set_base_color(np.power(color, 2.2))
                    else:
                        pass
    return robot

    # robot_builder = loader.load_file_as_articulation_builder(filename)
    # if disable_self_collision:
    #     for link_builder in robot_builder.get_link_builders():
    #         link_builder.set_collision_groups(1, 1, 17, 0)
    # else:
    #     if "allegro" in robot_name:
    #         for link_builder in robot_builder.get_link_builders():
    #             if link_builder.get_name() in ["link_9.0", "link_5.0", "link_1.0", "link_13.0", "base_link"]:
    #                 link_builder.set_collision_groups(1, 1, 17, 0)


def add_left():
    loader = scene.create_urdf_loader()
    left_builder: sapien.ArticulationBuilder = loader.load_file_as_articulation_builder("robot/svh_hand_right.urdf")
    for link_builder in left_builder.get_link_builders():
        link_builder.set_collision_groups(1, 1, 17, 0)

    left = left_builder.build(fix_root_link=True)
    left.set_root_pose(sapien.Pose([0.0, 0, 0.2], [0.707, 0.707, 0, 0]))
    left.set_qpos(left.get_qpos()*0)
    return modify_robot_visual(left)


def render(camera):
    scene.update_render()
    camera.take_picture()
    rgb = camera.get_color_rgba()
    return (rgb * 255).astype(np.uint8)


camera = add_camera_light(light=True)
camera2 = add_camera_light(light=False)
left = add_left()

import tqdm
outputs = []

groups = {}
idx = 0
group_id = {}
for i in left.get_joints():
    if i.get_dof() > 0:
        k = i.name.split('_')[1][:2]
        if k not in groups:
            groups[k] = []
        group_id[idx] = k
        groups[k].append(idx)
        idx += 1

print(groups)
np.random.seed(0)

aa = None


def sample_unit_ball_north_():
    xyz = np.random.randn(3)
    if xyz[2] < 0:
        xyz[2] *= -1
    if xyz[0] > 0:
        xyz[0] *= -1
    return xyz / np.linalg.norm(xyz)

from mani_skill2.utils.sapien_utils import get_entity_by_name, look_at
import os
os.makedirs('hand', exist_ok=True)

scene_center = np.array([0.0,0.0,0.2])
pose = look_at(0.5*np.array([0., 0., -1.])+scene_center, scene_center)
camera2.set_local_pose(pose)

def change_camera():
    # scene_center = np.array([-0.2,0.,0.2])
    # radius = 1.0
    radius = 0.5
    pose = look_at(radius*sample_unit_ball_north_()+scene_center, scene_center)
    camera.set_local_pose(pose)

    pose_new = pose.to_transformation_matrix()
    r = pose_new[:3,:3] @np.array([[0,-1,0],[0,0,1],[-1,0,0]]).T
    pose_new[:3,:3] = r
    return pose, pose_new


from mani_skill2.agents.camera import get_camera_images

output = {'image': [], 'pose': [], 'depth': [], 'actor_seg': [], 'camera_pose': []} 




output['intrinsic'] = camera.get_intrinsic_matrix()
all_points = []
for i in tqdm.trange(100):
    # if i % 10 == 0:
    #     rand_actions = np.random.normal(size=left.get_qf().shape) * 0.3
    #     rand_actions[-5:] *= 0.03 #THUMB
    # rand_actions = rand_actions * 0.
    pf = left.compute_passive_force(

                    gravity=True, 
                    coriolis_and_centrifugal=True, 
        
    )
    rand_actions = np.random.normal(size=left.get_qf().shape) * 0.3
    # aa = aa * 0.9 + rand_actions * 0.1 if aa is not None else rand_actions
    # if i < 50:
    #     #for j in groups:
    #     #    rand_actions[groups[j]] = 0
    #     for j in group_id:
    #         if group_id[j] == 'WR' or group_id[j] == names[j//10]:
    #             rand_actions[j] = 0
    left.set_qf(rand_actions + pf)
    for i in range(10):
        scene.step()

    _, camera_pose = change_camera()
    img = render(camera)



    ret = get_camera_images(camera, rgb=False, depth=True, visual_seg=True, actor_seg=True)
    output['image'].append(img)
    output['depth'].append(ret['depth'])
    output['actor_seg'].append(ret['actor_seg'].astype(np.uint8)[..., 0])
    output['camera_pose'].append(camera_pose)

    # output test
    img2 = render(camera2)
    outputs.append(img2)

    # o, d = get_rays(intr, c2w)
    # o = o.reshape(512, 512, 3)
    # d = d.reshape(512, 512, 3)
    # print(o.shape, d.shape)
    # print(o[256, 256] + d[256, 256] * 0.5)
    # exit(0)

    @torch.no_grad()
    def get_rays(intrinsic, c2w):
        W = 512
        H = 512
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) 
        i, j = i.t(), j.t()
        i, j = i+0.5, j+0.5

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0,-1]
        cy = intrinsic[1,-1]

        dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d, dirs

    c2w = torch.tensor(camera_pose, dtype=torch.float32)
    intr = torch.tensor(camera.get_intrinsic_matrix(), dtype=torch.float32)
    o, d, dirs = get_rays(intr, c2w)
    dirs = dirs.reshape(-1, 3)

    depth = ret['depth'].reshape(-1, 1)
    dirs = dirs * depth
    dirs = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)

    pcd = o + dirs
    pcd = pcd[depth[..., 0] > 0.]
    all_points.append(pcd.detach().cpu().numpy())

import open3d as o3d
p = o3d.geometry.PointCloud()
pcd = np.concatenate(all_points, 0)
print(pcd.shape, pcd.min(axis=0), pcd.max(axis=0))
p.points = o3d.utility.Vector3dVector(pcd)
o3d.io.write_point_cloud("hand/copy_of_fragment.pcd", p)


    #print(depth.min(), depth.max())



import torch
def plot(traj, output_path):
    np.savetxt(os.path.join(output_path, f'0_intrinsic.txt'), traj['intrinsic'])
    for idx, img in enumerate(traj['image']):
        #alpha = ((img == 170).sum(axis=2) != 3).astype(np.uint8) * 255
        #img = np.concatenate((img, alpha[:,:,None]), axis=2)
        img_name = f"0_{idx:03d}.png"
        im.fromarray(img).save(os.path.join(output_path, img_name))

        np.savetxt(os.path.join(output_path, f'0_{idx:03d}.txt'), traj['camera_pose'][idx])

        img_name = f"0_{idx:03d}_depth.npy"
        np.save(os.path.join(str(output_path), img_name), traj['depth'][idx])

        img_name = f"0_{idx:03d}_seg.png"
        im.fromarray(traj['actor_seg'][idx]).save(os.path.join(str(output_path), img_name))


        img_name = f"0_{idx:03d}_seg_c.png"
        import matplotlib.pyplot as plt
        colors = np.array(plt.get_cmap('tab20').colors)
        colors = colors[traj['actor_seg'][idx] % 20]
        im.fromarray((colors * 255).astype(np.uint8)).save(os.path.join(str(output_path), img_name))

plot(output, 'hand')

from tools.utils import animate
animate(outputs, 'hand/test.mp4')