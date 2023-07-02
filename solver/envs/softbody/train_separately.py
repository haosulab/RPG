import torch
import gym
import numpy as np
from torch import nn
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_pickup import PickUp
from gym.spaces import Discrete
from solver.rpg_net import Actor


class Pick3Cube(PickUp):

    def __init__(self, cfg=None, pad_dir_weight=0.75, write_text=False, boxid=None, reach_reward=1., norm='l2', restitution_coeff=0.1, low_steps=120):
        super().__init__(
            cfg, 
            dt=0.005, frame_skip=0, gravity=-1.0,
            cam_resolution=(512, 512),
            A_ACT_MUL=0.01, X_OBS_MUL=2.0, V_OBS_MUL=1.0
        )
        self.pad_dir_weight = pad_dir_weight
        MULT = 1.
        self.A_ACT_MUL = torch.tensor(
            # [1 / 100 /MULT, 1 / 100 /MULT, 1 / 100 /MULT, 1 / 100, 1 / 100]
            # [1 / 200, 1 / 200, 1 / 200, 1 / 100, 1 / 200, 1 / 100, 1 / 100]
            [1/40, 1/40, 1/40, 1/30, 1/30]
        )

        self.observation_space = gym.spaces.Box(-1, 1, (self.sim.world.getStateSize()+1, ))
        #self.sim.viewer.set_camera_rpy(r=0.5)#, p=-np.arctan2(2, 4), y=0)
        #self.sim.viewer.window.set_camera_rpy(r=0.5, p=-np.arctan2(2, 4), y=0)
        from tools.utils import lookat
        import sapien.core as sapien
        import transforms3d
        distance = 3.
        self.sim.viewer.camera_mount_actor.set_pose(
            sapien.Pose(
                [-1.8, -2 * distance + 1.5, distance - 0.3],
                # transforms3d.euler.euler2quat(0.0, np.arctan2(2, 4), np.pi / 2)
                transforms3d.euler.euler2quat(0.0, np.arctan2(2, 4), np.pi / 2 - 0.5)
            )
        )

    def init_simulator(self):
        # arm
        self.sim.arm = self.sim.load_urdf(
            "gripper_v3.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff,
            mass=1, inertia=[10., 10., 10.])
        
        # boxes
        self.sim.box = [
            self.sim.load_urdf(
                "sphere.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1]),
            self.sim.load_urdf(
                "sphere.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1]),
            self.sim.load_urdf(
                "sphere2.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1])
        ]

        # ground
        self.sim.load_urdf(
            "ground.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff)
        
        # action only control arm, not other objects
        for i in range(self.sim.arm.sapien_actor.dof, self.sim.world.getActionSize()):
            self.sim.world.removeDofFromActionSpace(i)
        
        # nimble arm forward kinematics
        self.sim.arm_fk = nimble.neural.IKMapping(self.sim.world)
        self.sim.arm_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("end_effector"))

        # nimble pad forward kinematics
        self.sim.pad_fk = nimble.neural.IKMapping(self.sim.world)
        self.sim.pad_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("left_pad"))
        self.sim.pad_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("right_pad"))
    
    def sample_state_goal(self, batch_size=1):
        box_x = 0.4
        box_y = (1 - box_x ** 2) ** 0.5
        box_y *= 0.7

        state = torch.tensor(
            [

            # arm qpos
            0.0, 0., 0.8, 0, 0,

            # 0, 0, 0,      0.0, -0.399,   -0.0,#            -1.5,
            0, 0, 0,      -0.4, -0.399,   -0.52,# -(1.65 - box_x),
            # 0, 0, 0,      -0.10, -0.399,  -0.4,# -(1.65 - box_x),
            0, 0, 0,      0.02, -0.399,  0.,# -(1.65 - box_x),
            0, 0, 0,      -0.4, -0.399,   0.52,# -(1.65 - box_x),
            # 0, 0, 0, 0, -0.399,      0,
            # 0, 0, 0, 0, -0.399,  box_y,
            # 0, 0, 0, 0, -0.399, -box_y,
            # velocities
            0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]
        )
        goals = torch.tensor(
            [
                     0, 0.4, 1.0,
                 box_y, 0.4, 1.0,
                -box_y, 0.4, 1.0,
            ]
        )
        return state, goals
    
    def get_obs(self, state=None):
        obs = super().get_obs(state)
        b, c = obs.shape
        k = obs.new(b, c+1)
        k[:, :-1] = obs
        k[:, -1] = self.t < 60
        return k
        

    def box_pos(self, state, box_index):
        assert 0 <= box_index < len(self.sim.box)
        return state[5 + 6 * (box_index) + 3: 5 + 6 * (box_index) + 6].clone()

    def step(self, action):
        from tools.utils import clamp
        #return super().step(clamp(action, -1., 1.))
        return super().step(action)

    def get_reward(self, s, a, s_next):
        # pos of boxes

        # left should be larger, right should be less than 0

        boxes_s_next = torch.stack([
            self.box_pos(s_next, i) 
            for i in range(len(self.sim.box))
        ])
        # ee pos
        # ee_s_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) \
        #     + torch.tensor([0, -0.4, 0])
        # end_effector: nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next).view(-1, 3)
        # gripper: nimble.map_to_pos(self.sim.world, self.sim.pad_fk, s_next).view(-1, 3)
        state_v = s_next.reshape(2, -1)
        up_v = state_v[1][3]
        ee_s_next = nimble.map_to_pos(self.sim.world, self.sim.pad_fk, s_next).view(-1, 3).mean(dim=0)\
            + torch.tensor([0, -0.5, 0])

        true_ee =  nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next).view(3) + torch.tensor([0, -0.3, 0])
        ee_s_next = true_ee
        # select closest box
        # boxi = (ee_s_next - boxes_s_next).norm(dim=1).min(dim=0)[1]
        boxi = self._cfg.boxid
        # self.text += f"\naction: {arr_to_str(a * self.A_ACT_MUL)}"
        # self.text += f"\nselected box: {boxi}"
        # # pad facing
        ee_id = 2
        # self.text += f"\npad angle: {pad_angle}"
        # self.text += f"\nee pos: {arr_to_str(ee_s_next.detach().numpy())}"
        # rewards
        #reach_top = -(ee_s_next + torch.tensor([0, -0.2, 0]) - box_s_next).norm() ** 2
        #reach = -(ee_s_next - box_s_next).abs().sum()/3 # ** 2
        # gripper_facing = (pad_dir * ref_dir).sum() ** 2
        # print(ee_s_next, s_next[ee_id+1], s_next[ee_id+2])
        gripper_qpos_at_0 = -(s_next[ee_id + 1] ** 2 + s_next[ee_id + 2] ** 2)
        gripper_close = -((s_next[ee_id + 1] - 0.5) ** 2 + (s_next[ee_id + 2] + 0.5) ** 2)
        gripper_ground_penalty = ((ee_s_next[1] - 0.2 + 0.4) < 0) * (-(ee_s_next[1] - 0.2 + 0.4) ** 2)

        #boxi = 1
        #boxi=0
        def mynorm(x, method=None):
            method = method or self._cfg.norm
            if method == 'l2':
                return -x.norm() ** 2 #maximize
            elif method == 'l1':
                return -x.abs().mean()
            elif method == 'exp_l2': # maximie
                return torch.exp(-x.norm()/0.2) + mynorm(x, 'l2')
            elif method == 'exp_l1': # maximie
                return torch.exp(-x.norm()/0.2) + mynorm(x, 'l1')
            else:
                raise NotImplementedError

        #ee_s_next = true_ee
        #if self.t < 70:
            # boxes_s_next[:, 1] = 0.21
        
        if boxi is None:

            reach = []
            dists = []
            for i in range(3):
            #for i in [boxi]:
                dists.append((boxes_s_next[i] - ee_s_next).norm())
                reach.append(mynorm(boxes_s_next[i] - ee_s_next))
            reach = torch.stack(reach).max(axis=0)[0]
            boxi = 2
        else:
            dist = (boxes_s_next[boxi] - ee_s_next).norm()
            reach = mynorm(boxes_s_next[boxi] - ee_s_next)

        box_s_next = boxes_s_next[boxi]
        goal = self.goals.view(-1, 3)[boxi]


        if self.t < 90:
            if self.t == 89:
                res = dict(
                    # height = -torch.relu(-(ee_s_next[1] - 0.2)) * 40, #- torch.log(torch.relu(ee_s_next[1] - 0.2)/0.01) # log barrier,
                    # reward_gripper_center_deviate=gripper_center_deviate * 20,
                    reward_gripper_qpos=gripper_qpos_at_0 * 20,
                    # gripper_ground_penalty=gripper_ground_penalty * 10,
                    reward_reach=reach * self._cfg.reach_reward * 10,
                )
                print(self.t, ee_s_next, res)
            else:
                res = dict(reward_reach=reach * 0., reward_gripper_qpos=gripper_qpos_at_0 * 20)
        # elif self.t < 50:
        #     res = dict(
        #         reward_reach=reach * self._cfg.reach_reward,
        #         # reward_gripper_facing=-(pad_angle ** 2 * self.pad_dir_weight),#gripper_facing * self.pad_dir_weight, # 0.75 is good
        #         reward_gripper_center_deviate=gripper_center_deviate * 20,
        #         reward_gripper_qpos=gripper_qpos_at_0 * 20,
        #         gripper_ground_penalty=gripper_ground_penalty * 10,
        #     )
        #     if self.t == 49:
        #         print('49', ee_s_next, res)
        else:
            res = dict(
                reward_reach=reach * 1 * self._cfg.reach_reward,
                reward_box_pull=(box_s_next[1] -0.) * 4 + (box_s_next[1] > -0.3) * 10, #* torch.exp(-dist/0.06), #-((box_s_next - goal) ** 2).sum() * 10,
                # reward_gripper_facing=-(pad_angle ** 2 * self.pad_dir_weight),
                # reward_gripper_center_deviate=gripper_center_deviate * 20,
                reward_gripper_qpos=gripper_close * 20,
                # gripper_ground_penalty=gripper_ground_penalty * 10,

                #reward_up = torch.relu(5.-up_v) * 1. if self.t > 100 else 0.,
                # reward_up = (ee_s_next[1] - 0.3).clamp(-100, 0.) * 0.01,
            )
            if self.t == self._cfg.low_steps-1:
                print(self._cfg.low_steps-1, ee_s_next, res, up_v, box_s_next)
        # if self.t == 49 or self.t == 99:
        #     print("here")
        #     print(res)
        return res



def main():
    n_agent = 1
    lr = 0.01
    n_epoch = 1000
    n_batch = 30# 100
    mutual_decay = 0.1
    from tools.utils import logger
    logger.configure(dir='tmp')

    env = Pick3Cube(reach_reward=1., n_batches=1, low_steps=120) #, boxid=2
    obs_space = env.observation_space
    action_space = env.action_space

    actors = []

    a_head = dict(
        TYPE='Normal',
        linear=True,
        squash=False,
        std_mode='fix_no_grad',
        std_scale=0.008
    )


    for i in range(n_agent):
        actors.append(
            Actor(
            obs_space,
            action_space,
            head=a_head, timestep=120,
        ).cuda()
    )

    info_net = Actor(obs_space, Discrete(n_agent), head=dict(TYPE='Discrete', epsilon=0.0)).cuda()
    params = list(nn.ModuleList(actors).parameters())

    actor_optim = torch.optim.Adam(params, lr=lr)
    info_optim = torch.optim.Adam(nn.ModuleList([info_net]).parameters(), lr=lr)

    # images = []
    # env.reset()
    # for i in range(100):
    #     a = env.action_space.sample()
    #     a*=0
    #     a[4] = 1.
    #     if i == 0 or i == 100:
    #         print(env.get_state()[:7])
    #     from tools.utils import totensor
    #     env.step(totensor(a, 'cpu'))
    #     print(env.get_state()[4])
    #     images.append(env.render('rgb_array'))
    # logger.animate(images, 'tmp.mp4')
    # exit(0)


    def run(k, return_image=False):
        policy = actors[k]
        obs = env.reset()
        total = 0
        if return_image:
            images = [env.render(mode='rgb_array')]

        traj = []
        for t in range(env.low_steps):
            p_a = policy(obs, timestep=t)
            a = p_a.rsample()[0]
            obs, r, _, _ = env.step(a.cpu())
            total += r
            traj.append(obs)

            if return_image:
                images.append(env.render(mode='rgb_array'))
            
        output = {
            'reward': total,
            'traj': torch.stack(traj),
        }
        if return_image:
            output['image'] = images
        return output

    #output = run(0, return_image=True)
    #logger.animate(output['image'], 'traj.mp4')
    def eval():
        images = [[] for k in range(env.low_steps+1)]
        for k in range(n_agent):
            output = run(k, return_image=True)
            for idx, j in enumerate(output['image']):
                images[idx].append(j)
        images = [np.concatenate(i, 1) for i in images]
        logger.animate(images, 'traj.mp4')



    mutual_weight = 1.
    import tqdm
    for i in range(n_epoch):
        avg_reawrds = [0 for i in range(n_agent)]
        for batch_id in tqdm.trange(n_batch):
            actor_optim.zero_grad()

            total_reward = 0

            info_data = []
            info_k = []
            rewards = []
            for k in range(n_agent):
                print("Actor", k)
                output = run(k)
                total_reward += output['reward'].sum()
                info_data.append(output['traj'].cuda())
                info_k.append(torch.zeros(*output['traj'].shape[:2], dtype=torch.long).cuda() + k)
                avg_reawrds[k] += output['reward'].item()
                rewards.append(output['reward'].item())

            info_data = torch.concat(info_data, axis=1)
            info_k = torch.concat(info_k, axis=1)

            logp = info_net(info_data).log_prob(info_k).sum()
            total_reward = total_reward + logp * mutual_weight

            (-total_reward).backward()
            actor_optim.step()

            nn.utils.clip_grad_norm_(params, 0.5)
        
            info_optim.zero_grad()
            logp = info_net(info_data.detach()).log_prob(info_k.detach()).sum()
            logp.backward()
            info_optim.zero_grad()
            print('rewards', rewards)
        print('avg_rewards', [i/n_batch for i in avg_reawrds])
        eval()

        mutual_weight = mutual_weight * mutual_decay


if __name__ == '__main__':
    main()