import torch
from torch_geometric.nn import Linear

class GeodLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, geodesic_distances, initial_scaling_value):
        super(GeodLinear, self).__init__()
        self.geodesic_distances = geodesic_distances.unsqueeze(1)
        self.scaling_value = torch.nn.Parameter(initial_scaling_value)
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        return (1 - torch.exp(-self.scaling_value * self.geodesic_distances)) * self.linear(x)