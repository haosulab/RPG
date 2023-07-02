from inspect import isfunction
import torch
import numpy as np
import torch.multiprocessing as multip

class Worker(multip.Process):
    RESET = 1
    STEP = 2
    RENDER = 3
    GET = 4
    EXIT = 5

    def __init__(self, cls, *args, **kwargs):
        multip.Process.__init__(self)
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.pipe, self.worker_pipe = multip.Pipe()
        self.daemon = True
        self.start()

    def wait(self):
        self.pipe.recv()

    def run(self):
        #env = self.create_fn[0](*self.create_fn[1:])
        if isfunction(self.cls):
            func = self.cls
        else:
            func = self.cls(*self.args, **self.kwargs)
        self.worker_pipe.send('START')

        ans = None
        while True:
            op, data = self.worker_pipe.recv()
            if op == self.RESET:
                ans = func.reset(*data)
                self.worker_pipe.send(ans)

            elif op == self.STEP:
                ans = func.step(*data)
                self.worker_pipe.send(ans)

            elif op == self.RENDER:
                ans = func.render(*data)
                self.worker_pipe.send(ans)

            elif op == self.EXIT:
                self.worker_pipe.close()
                return

    def reset(self, idx=None):
        self.pipe.send([self.RESET, [idx]])
        return self.pipe.recv()

    def step(self, action):
        self.pipe.send([self.STEP, [action]])
        return self.pipe.recv()

    def render(self, mode, *args):
        self.pipe.send([self.RENDER, [mode, *args]])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()