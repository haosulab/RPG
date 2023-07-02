import os
from torch.utils.tensorboard import SummaryWriter as TorchWriter


class SummaryWriter:
    def __init__(self, path:str, writer=None, prefix=None):
        if path is not None:
            if not path.endswith("summary"):
                path = os.path.join(path, 'summary')
            if writer is None:
                writer = TorchWriter(log_dir=path)

        self.path = path
        self.writer = writer
        self.prefix = prefix
        self.steps = 0
        self.log_interval = 1

    def set_interval(self, interval=1):
        self.log_interval = interval

    def write(self, values, step=0):
        assert self.path is not None, "You can't write to a None path!"
        if self.steps % self.log_interval == 0:
            for key, val in values.items():
                if self.prefix is not None:
                    key = f"{self.prefix}/{key}"
                if val.squeeze().ndim == 0:
                    self.writer.add_scalar(key, val, step)
                else:
                    self.writer.add_images(key, val, step)
        self.steps += 1

    def new(self, prefix):
        if self.prefix is not None:
            prefix = f"{self.prefix}/{prefix}"
        return SummaryWriter(self.path, self.writer, prefix)
