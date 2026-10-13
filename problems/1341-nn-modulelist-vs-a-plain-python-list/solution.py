import torch
import torch.nn as nn


class LayerStack(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # TODO: build one nn.Linear(dims[i], dims[i+1]) per consecutive pair and
        #       store them in an nn.ModuleList assigned to self.layers

        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) 
            for i in range(len(dims) - 1)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        # TODO: apply each layer in order, with relu between consecutive layers
        #       but NOT after the final layer
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < len(self.layers) - 1:
                x = self.relu(x)
        return x
