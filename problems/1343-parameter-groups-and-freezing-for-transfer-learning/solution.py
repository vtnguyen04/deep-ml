import torch
import torch.nn as nn
import torch.optim as optim

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_optimizer(model, base_lr, head_lr, freeze_backbone):
    # TODO: when freeze_backbone is True, set requires_grad = False on every
    #       backbone parameter and return SGD over the head only, at head_lr
    # TODO: otherwise return SGD with two param groups:
    #       backbone at base_lr first, then head at head_lr
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

        param_groups = [
            {"params": model.head.parameters(), "lr": head_lr}
        ]
    else:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": base_lr},
            {"params": model.head.parameters(), "lr": head_lr}
        ]

    return optim.SGD(param_groups)