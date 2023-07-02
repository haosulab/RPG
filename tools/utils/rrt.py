import argparse
import numpy as np
import os
import pickle
import math
import tqdm

TRAPPED = 0
ADVANCED = 1
REACHED = 2


def steerTo(start, end, collision_check, step_size, expand_dis=np.inf):
    delta = end - start
    length = np.linalg.norm(delta)

    if length > 0:
        delta /= length
        length = min(length, expand_dis)
        for i in range(0, int(math.floor(length/step_size))):
            cur = start + (i * step_size) * delta
            if collision_check(cur):
                return 0, None
        end = start + length * delta
        if collision_check(end):
            return 0, None
    return 1, end


# checks the feasibility of path nodes only
def collision_check(path, collHandle):
    for i in range(0, len(path)):
        if collHandle(path[i]):
            return 0
    return 1


#lazy vertex contraction
def lvc(path, collHandle, steerTo, step_size):
    for i in range(0, len(path)-1):
        for j in range(len(path)-1, i+1, -1):
            if steerTo(path[i], path[j], collHandle, step_size)[0]:
                return lvc(path[0:i+1] + path[j:len(path)], collHandle, steerTo, step_size)
    return path


class Tree:
    def __init__(self, collision_check, expand_dis, step_size):
        self.nodes = []
        self.father = []
        self.path = []
        self.collision_check = collision_check
        self.expand_dis = expand_dis
        self.step_size = step_size

    def is_reaching_target(self, start1, start2):
        return np.abs(start1 - start2).max() < self.step_size

    def extend(self, q):
        nearest_ind = self.get_nearest_node_index(self.nodes, q)
        nearest_node = self.nodes[nearest_ind]
        flag, new_node = steerTo(nearest_node, q, self.collision_check, step_size=self.step_size, expand_dis=self.expand_dis)
        if flag:
            self.add_edge(new_node, nearest_ind)
            if self.is_reaching_target(new_node, q):
                return REACHED, new_node
            else:
                return ADVANCED, new_node
        else:
            return TRAPPED, None

    def connect(self, q):
        while True:
            S = self.extend(q)[0]
            if S != ADVANCED:
                break
        return S

    def add_edge(self, q, parent_id):
        self.nodes.append(q)
        self.father.append(parent_id)
        if parent_id != -1:
            self.trees.append([q, self.nodes[parent_id]])

    def backtrace(self):
        cur = len(self.nodes) - 1
        path = []
        while cur != 0:
            path.append(self.nodes[cur])
            cur = self.father[cur]
        return path


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        #TODO K-D tree or OCTTree
        dlist = [math.sqrt(((node - rnd_node) ** 2).sum()) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind


class RRTConnectPlanner:
    def __init__(self,
                 state_sampler,
                 collision_checker,
                 expand_dis=0.1,
                 step_size=0.01,
                 max_iter=0,
                 use_lvc=False):

        self.state_sampler = state_sampler
        self.collision_checker = collision_checker
        self.expand_dis = expand_dis
        self.step_size = step_size
        self.max_iter = max_iter
        self.use_lvc = use_lvc

    def __call__(self, start, goal, info=False, return_tree=False, rrt_connect=True):
        start = np.array(start)
        goal = np.array(goal)
        self.TA = self.TB = None

        # code for single direction
        self.TA = TA = Tree(self.collision_checker, self.expand_dis, self.step_size)

        trees = []
        TA.trees = trees
        TA.add_edge(start, -1)

        self.TB = TB = Tree(self.collision_checker, self.expand_dis, self.step_size)
        TB.trees = trees
        TB.add_edge(goal, -1)

        ran = range if not info else tqdm.trange
        for i in ran(self.max_iter):
            q_rand = np.array(self.state_sampler())
            S, q_new = TA.extend(q_rand)
            if not (S == TRAPPED):
                reached = (rrt_connect and TB.connect(q_new) == REACHED)
                if not rrt_connect and np.linalg.norm(q_new - goal) < self.expand_dis:
                    reached = steerTo(q_new, goal, self.collision_checker, step_size=self.step_size, expand_dis=self.expand_dis)[0]
                if reached:
                    if i % 2 == 1 and rrt_connect:
                        TA, TB = TB, TA
                        self.TA = TA
                        self.TB = TB

                    path = TA.backtrace()[::-1] + TB.backtrace()[1:]
                    path = [start] + path + [goal]
                    print('last one in path', path[-1], 'length', len(path), 'lvc...', self.use_lvc)
                    if self.use_lvc and len(path) <= int(self.use_lvc):
                        self.original_path = path
                        path = lvc(path, self.collision_checker, steerTo, self.step_size)
                    if return_tree:
                        return path, trees
                    return path
            if rrt_connect:
                TA, TB = TB, TA
        return []