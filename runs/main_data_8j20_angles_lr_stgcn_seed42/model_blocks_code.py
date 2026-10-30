"""Small, randomly initialized ST-GCN proposal. No training entry point."""
import torch
from torch import nn

# Local nodes 0..11 correspond to COCO keypoints 5..16.
EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
         (0, 6), (1, 7), (6, 7), (6, 8), (8, 10),
         (7, 9), (9, 11))
LOWER_BODY_EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5))


def make_adjacency(num_nodes=12, edges=EDGES, roots=(6, 7)):
    """Partitions: equal hip distance, closer source, farther source.

    A[k, destination, source] aggregates source features at destination.
    Each partition is normalized by its incoming degree, independently.
    """
    neighbors = [set() for _ in range(num_nodes)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    distance = [99] * num_nodes
    queue = list(roots)
    for root in roots:
        distance[root] = 0
    for node in queue:
        for neighbor in sorted(neighbors[node]):
            if distance[neighbor] == 99:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    adjacency = torch.zeros(3, num_nodes, num_nodes)
    for dst in range(num_nodes):
        for src in sorted(neighbors[dst] | {dst}):
            partition = (0 if distance[src] == distance[dst]
                         else 1 if distance[src] < distance[dst] else 2)
            adjacency[partition, dst, src] = 1
    return adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.out_channels = out_channels
        self.spatial = nn.Conv2d(in_channels, 3 * out_channels, 1, bias=False)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (5, 1),
                      stride=(stride, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout),
        )
        self.residual = (nn.Identity() if in_channels == out_channels and stride == 1
                         else nn.Sequential(
                             nn.Conv2d(in_channels, out_channels, 1,
                                       stride=(stride, 1), bias=False),
                             nn.BatchNorm2d(out_channels)))
        self.activation = nn.ReLU()

    def forward(self, x, adjacency):
        n, _, t, v = x.shape
        spatial = self.spatial(x).reshape(n, 3, self.out_channels, t, v)
        spatial = torch.einsum('nkctv,kwv->nctw', spatial, adjacency)
        return self.activation(self.temporal(spatial) + self.residual(x))


class PenaltySTGCN(nn.Module):
    """Input (N, 4, 16, 12, 1); output logits (N, 3), L/C/R."""
    def __init__(self, dropout=0.3, num_joints=12, edges=EDGES, roots=(6, 7),
                 num_classes=3):
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer('adjacency', make_adjacency(num_joints, edges, roots))
        self.input_norm = nn.BatchNorm1d(4 * num_joints)
        self.blocks = nn.ModuleList([
            STGCNBlock(4, 32, dropout=dropout),
            STGCNBlock(32, 32, dropout=dropout),
            STGCNBlock(32, 64, stride=2, dropout=dropout),
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (4, 16, self.num_joints, 1):
            raise ValueError(f'Expected input shape (N, 4, 16, {self.num_joints}, 1)')
        n = x.shape[0]
        x = x[..., 0].permute(0, 3, 1, 2).reshape(n, 4 * self.num_joints, 16)
        x = self.input_norm(x).reshape(n, self.num_joints, 4, 16).permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.head(x.mean(dim=(2, 3)))
