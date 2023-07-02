#from envs
#from envs.triple_ant import TripleAntEnv
#env = TripleAntEnv()
from envs.mp_envs import make

from generative.vae_trainer import RandomTrainer

env = make('LargeMaze')
random = RandomTrainer.parse(env)

random.start()
