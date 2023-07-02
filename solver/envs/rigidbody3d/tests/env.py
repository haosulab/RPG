import torch
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_grasp import GraspBox

class TestPullUp(GraspBox):

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def init_simulator(self):

        self.sim.world.setGravity([0, -1, 0])

        # arm
        self.sim.arm    = self.sim.load_urdf(
            "gripper_v2.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff,
            mass=1)
        
        # boxes
        self.sim.box = [
            self.sim.load_urdf(
                "sphere.urdf", 
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

    def sample_state_goal(self, batch_size=1):
        state, goals = super().sample_state_goal(batch_size)
        state[3] += 0.2
        goals = torch.tensor([0, 0.5, 1.5])
        return state, goals

    def get_reward(self, s, a, s_next):
        r_mult = 0.1
        ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) \
            + torch.tensor([0, -0.4, 0])
        ee_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) \
            + torch.tensor([0, -0.4, 0])
        box = self.box_pos(s, 0)
        box_next = self.box_pos(s, 0)
        reach = -(ee_next - box_next).norm() ** 2

        gripper_center_deviate = -(s_next[5] - s_next[6]) ** 2 * 2
        gripper_qpos_at_0 = -(s_next[5] ** 2 + s_next[6] ** 2)
        gripper_close = -((s_next[5] - 0.12) ** 2 + (s_next[6] - 0.12) ** 2)

        # slip penalty
        slip_penalty = -((ee_next - ee) - (box_next - box)).norm() ** 2
        
        if self.t < 20:
            return (
                reach + gripper_qpos_at_0 + gripper_center_deviate
            )
        elif self.t < 40:
            return (
                reach + gripper_close + gripper_center_deviate + slip_penalty
            )
        else:
            box_to_goal = -(box - self.goals).norm() ** 2
            return (
                reach * 0.1 + gripper_close + gripper_center_deviate + slip_penalty + box_to_goal
            )# * r_mult * 5

if __name__ == "__main__":
    env = TestPullUp()
    obs = env.reset()

    env.sim.viewer.create_window()
    while not env.sim.viewer.window.closed:
        env.step(torch.tensor([0, 0, 0, -10, 0, 0, 0], dtype=torch.float32))
        # env.step(torch.randn((7, )))
        env.render()
        print(nimble.map_to_pos(env.sim.world, env.sim.arm_fk, env.sim.state).float())

