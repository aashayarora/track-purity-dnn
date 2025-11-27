import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackPurityDNN(nn.Module):
    def __init__(self, input_dim=45, hidden_dims=[256, 128, 64, 32], residual_dim=32, dropout=0.1, n_res_blocks=3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.initial_layers = nn.Sequential(*layers)
        
        self.fc_in = nn.Linear(prev_dim, residual_dim)
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(residual_dim, residual_dim),
                nn.ReLU(),
                nn.Linear(residual_dim, residual_dim),
                nn.ReLU(),
                nn.Linear(residual_dim, residual_dim),
                nn.ReLU(),
            ) for _ in range(n_res_blocks)
        ])
        
        self.out = nn.Linear(residual_dim, 1)
        
        self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial_layers(x)
        x_in = F.relu(self.fc_in(x))
        
        for block in self.res_blocks:
            res = block(x_in)
            x_in = x_in + res
        
        out = self.out(x_in)
        return out