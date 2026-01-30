import torch
from torch import nn


class TemporalEmbedding(nn.Module):
    def __init__(self, num_positions, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_positions, d_model)  # Learnable embeddings for time steps

    def forward(self, x):
        batch_size, temporal_dim, num_nodes, channels = x.shape  # Extract shape
        positions = torch.arange(temporal_dim, device=x.device).unsqueeze(0)  # Shape: (1, temporal_dim)
        embedded_positions = self.embedding(positions)  # Shape: (1, temporal_dim, d_model)
        # Expand to match (batch, nodes, temporal_dim, channels)
        embedded_positions = embedded_positions.unsqueeze(2).expand(batch_size, temporal_dim, num_nodes, channels)
        return x + embedded_positions
