import torch
import torch.nn as nn


class ScaledShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: register `scale` (ones of shape (dim,)) as a learnable parameter
        # TODO: register `shift` (zeros of shape (dim,)) as a learnable parameter
        # TODO: register `calls` as a BUFFER holding a scalar long tensor at 0


        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

        self.register_buffer("calls", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # TODO: if the module is in training mode, increment `calls` by 1
        # TODO: return x * scale + shift
        
        if self.training:
            self.calls += 1

        return x * self.scale + self.shift
