import torch
from torch import nn
import torch.nn.functional as F


def mean_weightening(hidden_states):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    return hidden_states.mean(dim=1)


def exponential_weightening(hidden_states, weight_scale):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    device = hidden_states.device

    indices = torch.arange(hidden_states.shape[1]).float().to(device)  # [0, 1, 2, ..., seq_len-1]
    weights = torch.exp(weight_scale * indices)  # Shape: [seq_len]

    # Normalize weights (optional, for scale invariance)
    weights = weights / weights.sum()

    # Reshape weights to [1, n_items, 1] for broadcasting
    weights = weights.view(1, hidden_states.shape[1], 1)

    # Apply weights and aggregate
    weighted_tensor = hidden_states * weights
    result = weighted_tensor.sum(dim=1)  # Aggregated tensor, shape: [batch_size, hidden_units]
    return result


class SimpleAttentionAggregator(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleAttentionAggregator, self).__init__()
        self.attention = nn.Linear(hidden_units, 1)  # Learnable attention weights

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, n_items, hidden_units]
        Returns:
        Aggregated tensor of shape [batch_size, hidden_units]
        """
        # Compute attention scores (shape: [batch_size, n_items, 1])
        scores = self.attention(x)

        # Normalize scores with softmax over the 2nd dimension
        weights = F.softmax(scores, dim=1)  # Shape: [batch_size, n_items, 1]

        # Weighted sum of the input tensor
        weighted_sum = (x * weights).sum(dim=1)  # Shape: [batch_size, hidden_units]
        return weighted_sum


# class NGNNAggregator(nn.Module):
#     def __init__(self, in_features, out_features, alpha=0.2):
#         super(NGNNAggregator, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha  # LeakyReLU negative slope
#
#         # Linear transformations
#         self.W = nn.Linear(in_features, out_features, bias=False)
#         self.attention = nn.Linear(2 * out_features, 1, bias=False)  # Attention mechanism
#
#         # Non-linearity
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h):
#         """
#         h: Input node features of shape [batch_size, n_items, hidden_units]
#         """
#         # Step 1: Linear transformation
#         z = self.W(h)  # Shape: [num_nodes, out_features]
#
#         # Step 2: Compute pairwise attention scores
#         num_nodes = z.size(0)
#         z_expanded = z.unsqueeze(0).repeat(num_nodes, 1, 1)  # Shape: [num_nodes, num_nodes, out_features]
#         z_concat = torch.cat([z_expanded, z_expanded.transpose(0, 1)], dim=-1)  # Shape: [num_nodes, num_nodes, 2*out_features]
#         e = self.leakyrelu(self.attention(z_concat)).squeeze(-1)  # Shape: [num_nodes, num_nodes]
#
#         # Step 3: Normalize scores with softmax
#         alpha = F.softmax(e, dim=1)  # Shape: [num_nodes, num_nodes]
#
#         # Step 4: Weighted aggregation
#         h_prime = torch.matmul(alpha, z)  # Shape: [num_nodes, out_features]
#
#         return h_prime, alpha