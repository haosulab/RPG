import tqdm
import numpy as np
from diffrl.env import DifferentiablePhysicsEnv

env = DifferentiablePhysicsEnv(observer_cfg={"TYPE": "ParticleObserver"})

state = env.get_state()

def loss_fn(idx, **kwargs):
    return kwargs['dist'].min(axis=0)[0].sum()

action = np.zeros((50, 12))

env.set_loss_fn(loss_fn)
env.set_initial_state_sampler(lambda env: state)

env.reset(requires_grad=True)
for i in tqdm.trange(100):
    grad = np.array(env.compute_gradient_example(state, np.concatenate((action, action)), loss_fn=loss_fn, max_episode_steps=50))
    print(grad.shape, np.linalg.norm(grad[:50]-grad[50:]) )
    action = action - grad[:len(action)] * 0.1

env.execute(state, action, 'xx.mp4')