import torch
from torch.nn import Softmax
from torch.nn.init import xavier_uniform_, zeros_

class SoftArgmax(torch.nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmax, self).__init__()
        self.beta = beta

    def forward(self, x):
        v = torch.exp(self.beta * x)
        return v/torch.sum(v, dim=1).unsqueeze(1)


class BeamMatchingSoftmaxMatrix(torch.nn.Module):
    def __init__(self, rows, columns):
        super(BeamMatchingSoftmaxMatrix, self).__init__()
        self.matching_matrix = torch.nn.Parameter(torch.zeros(rows, columns))
        self.softmax_layer = Softmax()
        self.softargmax_layer = SoftArgmax()
        # Initializing matching matrix
        # xavier_uniform_(self.matching_matrix)
        zeros_(self.matching_matrix)

    def forward(self, mode='softmax'):
        if mode == 'softmax':
            return self.softmax_layer(self.matching_matrix)
        elif mode == 'softargmax':
            return self.softargmax_layer(self.matching_matrix)
        else:
            raise ValueError('Inserted mode not valid.')