import os
import numpy as np
import time
import multiprocessing as multip
from multiprocessing import Process

class RenderFunc(Process):
    def __init__(self):
        multip.Process.__init__(self)
        self.input_queue = multip.Queue()
        self.output_queue = multip.Queue()
        self.daemon = True
        self.start()

    def run(self):
        while True:
            p = None
            while not self.input_queue.empty():
                p = self.input_queue.get()
            if p is not None:
                print('in render')
                time.sleep(1./20)
                self.output_queue.put(0)
            

def main():
    render_fn = RenderFunc()
    idx = 0
    import tqdm
    for i in tqdm.trange(1000000):
        time.sleep(1./20)
        render_fn.input_queue.put(np.zeros((1000, 3)))

        idx += 1
        if idx % 20 == 0:
            while not render_fn.output_queue.empty():
                k = render_fn.output_queue.get()
                print(k)

if __name__ == '__main__':
    main()