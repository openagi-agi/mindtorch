# mindtorch/core/mlp.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        """
        input_dim: int
        hidden_dims: list of ints, e.g., [128, 64]
        output_dim: int
        activation: activation function class (e.g. nn.ReLU, nn.Tanh)
        """
        super().__init__()
        dims = [input_dim] + hidden_dims

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())

        layers.append(nn.Linear(dims[-1], output_dim))  # Final output layer

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
