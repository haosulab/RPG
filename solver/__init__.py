# basic solver to solve the optimization problems
import os
SOLVER_PATH = os.path.dirname(os.path.abspath(__file__))

if 'DATA_PATH' not in os.environ:
    DATA_PATH = os.path.join(SOLVER_PATH, 'data')
else:
    DATA_PATH = os.environ['DATA_PATH']

if 'MODEL_PATH' not in os.environ:
    MODEL_PATH = os.path.join(SOLVER_PATH, 'model')
else:
    MODEL_PATH = os.environ['MODEL_PATH']