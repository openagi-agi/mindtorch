# mindtorch/core/recurrent/gru.py

import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len, input_dim)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden
