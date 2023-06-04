import torch
import torch.nn as nn


class Std(nn.Module):
    
    def __init__(
            self, out_dim, mode="local", learn=True, 
            scale=1, log_std_min=-20, log_std_max=2, init_log_std=0
        ):
        super().__init__()
        self.out_dim      = out_dim
        self.mode         = mode
        self.learn        = learn
        self.log_std_min  = log_std_min
        self.log_std_max  = log_std_max
        self.init_log_std = init_log_std
        self.scale        = scale
        assert self.mode in ["global", "local"], "mode must be 'global' or 'local'"
        
        # decide whether to manage std
        self.manage_std   = (mode == "global")
        if not self.learn:
            # if not learning std, then it's global
            self.manage_std = True
        
        # manage std
        self.log_std = None
        if self.manage_std:
            self.log_std = nn.Parameter(
                self.init_log_std * \
                torch.ones((1, out_dim), dtype=torch.float32)
            )
            self.in_dim = 0
        else:
            self.in_dim = out_dim

    def extra_repr(self):
        repr_str = (
            f"mode={self.mode},\n"
            f"learn={self.learn},\n"
            f"log_std_min={self.log_std_min},\n"
            f"log_std_max={self.log_std_max},\n"
            f"init_log_std={self.init_log_std},\n"
            f"scale={self.scale}")
        if self.manage_std:
            repr_str += f",\nlog_std={self.log_std.tolist()}"
        return repr_str

    def forward(self, x):
        if self.manage_std:
            # use the global logstd
            assert self.log_std is not None
            logstd = self.log_std
        else:
            # take the input as logstd
            assert x.shape[-1] == self.out_dim
            logstd = x + self.init_log_std
        
        logstd = torch.clamp(logstd, min=self.log_std_min, max=self.log_std_max)
        
        std = torch.exp(logstd)
        if not self.learn:
            # detach std if not learning it
            std = std.detach()
        
        return std * self.scale
