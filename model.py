"""Final central-camera six-joint ST-GCN (checkpoint-compatible)."""
import torch
from torch import nn

JOINTS = (11, 12, 13, 14, 15, 16)
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5))
FEATURES = tuple(f"kp_{joint}_{axis}n" for joint in JOINTS for axis in "xy")
LABELS = ("L", "R")

def make_adjacency(num_nodes=6, edges=EDGES, roots=(0, 1)):
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
    """Input (batch, 3, 12) normalized XY; output L/R logits."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.coordinate_norm = nn.BatchNorm1d(12)
        self.register_buffer("adjacency", make_adjacency())
        self.blocks = nn.ModuleList([
            STGCNBlock(2, 16, dropout=dropout),
            STGCNBlock(16, 16, dropout=dropout),
            STGCNBlock(16, 32, stride=2, dropout=dropout),
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, 12):
            raise ValueError("Expected (batch, 3 frames, 12 normalized XY features)")
        n, t, _ = x.shape
        coordinates = self.coordinate_norm(x.transpose(1, 2))
        coordinates = coordinates.reshape(n, 6, 2, t).permute(0, 2, 3, 1)
        for block in self.blocks:
            coordinates = block(coordinates, self.adjacency)
        return self.head(coordinates.mean(dim=(2, 3)))
