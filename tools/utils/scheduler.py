import abc
from tools.config import Configurable, as_builder


@as_builder
class Scheduler(Configurable, abc.ABC):
    def __init__(self, cfg=None, init=1.) -> None:
        super().__init__()

        self.epoch = 0
        self.init_value = init
        self.value = init

    def step(self, epoch=None):
        if epoch is None:
            self.epoch += 1
            delta = 1
        else:
            assert epoch > self.epoch
            delta = epoch - self.epoch
            self.epoch = epoch

        self.value = self._step(self.epoch, delta)
        return self.value
    
    def get(self):
        return self.value


    @abc.abstractmethod
    def _step(self, cur_epoch, delta):
        pass


class constant(Scheduler):
    def __init__(self, cfg=None) -> None:
        super().__init__()

    def _step(self, cur_epoch, delta):
        return self.init_value


class exp(Scheduler):
    def __init__(self, cfg=None, gamma=0.99, min_value=0., start=0, end=None, target_value=None) -> None:
        super().__init__()
        self.gamma = gamma
        self.min_value = min_value

        self.start = start
        self.end = end

        if self.end is not None:
            # decaying b
            assert self.end > self.start
            import numpy as np
            target_value = target_value or self.min_value
            self.gamma = np.exp((np.log(max(target_value, 1e-10)) - np.log(self.init_value))  / (self.end - self.start))

    def _step(self, cur_epoch, delta):
        if cur_epoch < self.start:
            return self.init_value

        if self.end is None:
            return max(self.min_value, self.value * (self.gamma ** delta))
        if cur_epoch > self.end:
            return self.value
        return max(self.min_value, self.init_value * (self.gamma ** (cur_epoch - self.start)))

class linear(Scheduler):
    def __init__(self, cfg=None, start=0, min_value=0., end=None, target_value=None, delta=None) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.delta = delta
        self.min_value = min_value

        if self.end is not None:
            # decaying b
            assert self.end > self.start
            import numpy as np
            target_value = target_value or self.min_value
            #self.gamma = np.exp((np.log(max(target_value, 1e-10)) - np.log(self.init_value))  / (self.end - self.start))
            self.delta = (target_value - self.init_value) / (self.end - self.start)

    def _step(self, cur_epoch, delta):
        if cur_epoch < self.start:
            return self.init_value

        if self.end is None:
            return max(self.min_value, self.value + self.delta * delta)
        if cur_epoch > self.end:
            return self.value
        return max(self.min_value, self.init_value + self.delta * (cur_epoch - self.start))


# class exp2(Scheduler):
#     def __init__(self, cfg=None, start=0, end=None, target=0.) -> None:
#         super().__init__()
#         self.start = start
#         assert end is not None, "please specify the ending state of exponetial"

#     def _step(self, cur_epoch, delta):
#         #return max(self.min_value, self.value * (self.gamma ** delta))
#         if cur_epoch < self.start:
#             return self.init_value

class stage(Scheduler):
    def __init__(self, cfg=None, milestones=None, gamma=0.1) -> None:
        super().__init__()
        self.milestones = milestones
        self.gamma = gamma

    def _step(self, cur_epoch, delta):
        value = self.init_value
        for i in self.milestones:
            if cur_epoch >= i:
                value = value * self.gamma
        return value