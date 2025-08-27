import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))  # final layer, no activation
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
