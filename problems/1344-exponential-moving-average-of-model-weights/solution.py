import torch
import torch.nn as nn


class EMA:
    def __init__(self, model, decay):
        # TODO: store decay and a detached clone of every named parameter
        self.decay = decay

        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        # TODO: shadow = decay * shadow + (1 - decay) * param, in place,
        #       without touching the live model
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0-self.decay)

    def copy_to(self, model):
        # TODO: write the shadow values into the model's parameters in place
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])

