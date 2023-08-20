import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.ll1 = nn.Linear(14, 1, dtype=torch.float32)

    def forward(self, x):
        return self.ll1(x)