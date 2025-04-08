import torch
from torch_geometric.nn.pool import knn

def repr_chamfer_distance(x, y, simmetrical=True):
    idxs = knn(y, x, 1)[1, :]
    dist1 = torch.sum(torch.norm(x - y[idxs], dim=1))

    if simmetrical:
        idxs = knn(x, y, 1)[1, :]
        dist2 = torch.sum(torch.norm(y - x[idxs], dim=1))
        return dist1 + dist2
    else:
        return dist1