import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import math
__all__ = ['MRN']

class MRN(nn.Module):
    def __init__(self, in_dim=1000, hid_dim=256, num_layers=1, cell='lstm'):
        super(MRN, self).__init__()
        self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p
    


