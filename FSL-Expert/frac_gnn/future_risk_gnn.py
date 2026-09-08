from __future__ import annotations

import torch
from torch import nn


def build_knowledge_adjacency(window_size: int, rule_node_count: int = 3) -> torch.Tensor:
    node_count = window_size + rule_node_count
    adjacency = torch.eye(node_count, dtype=torch.bool)
    for node in range(window_size - 1):
        adjacency[node, node + 1] = True
        adjacency[node + 1, node] = True
    for rule_offset in range(rule_node_count):
        rule_node = window_size + rule_offset
        for time_node in range(window_size):
            adjacency[rule_node, time_node] = True
            adjacency[time_node, rule_node] = True
    return adjacency


class DenseGraphAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.source_attention = nn.Linear(output_dim, 1, bias=False)
        self.target_attention = nn.Linear(output_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(nodes)
        source = self.source_attention(hidden)
        target = self.target_attention(hidden).transpose(1, 2)
        logits = self.activation(source + target)
        mask = adjacency.to(device=nodes.device).unsqueeze(0)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        return torch.bmm(weights, hidden)


class KnowledgeTemporalRiskGNN(nn.Module):
    def __init__(
        self,
        base_feature_dim: int,
        window_size: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        use_rule_nodes: bool = True,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.rule_node_count = 3 if use_rule_nodes else 0
        self.use_rule_nodes = use_rule_nodes
        self.context_dim = context_dim
        node_type_dim = 4 if use_rule_nodes else 1
        node_input_dim = base_feature_dim + node_type_dim + 1
        self.register_buffer(
            "adjacency",
            build_knowledge_adjacency(window_size, self.rule_node_count),
        )
        self.layer1 = DenseGraphAttention(node_input_dim, hidden_dim, dropout)
        self.layer2 = DenseGraphAttention(hidden_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _nodes(self, windows: torch.Tensor, rule_scores: torch.Tensor) -> torch.Tensor:
        batch_size, _, base_dim = windows.shape
        device = windows.device
        if not self.use_rule_nodes:
            time_type = torch.ones(batch_size, self.window_size, 1, device=device)
            activation = torch.zeros(batch_size, self.window_size, 1, device=device)
            return torch.cat([windows, time_type, activation], dim=-1)

        time_types = torch.zeros(batch_size, self.window_size, 4, device=device)
        time_types[:, :, 0] = 1.0
        time_activation = torch.zeros(batch_size, self.window_size, 1, device=device)
        time_nodes = torch.cat([windows, time_types, time_activation], dim=-1)

        rule_features = torch.zeros(batch_size, 3, base_dim, device=device)
        rule_types = torch.zeros(batch_size, 3, 4, device=device)
        for rule_index in range(3):
            rule_types[:, rule_index, rule_index + 1] = 1.0
        activation = rule_scores[:, :3].unsqueeze(-1)
        rule_nodes = torch.cat([rule_features, rule_types, activation], dim=-1)
        return torch.cat([time_nodes, rule_nodes], dim=1)

    def forward(
        self,
        windows: torch.Tensor,
        rule_scores: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nodes = self._nodes(windows, rule_scores)
        hidden = torch.relu(self.norm1(self.layer1(nodes, self.adjacency)))
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.norm2(self.layer2(hidden, self.adjacency)))
        time_hidden = hidden[:, : self.window_size]
        pooled = torch.cat([time_hidden.mean(dim=1), time_hidden[:, -1]], dim=-1)
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError(f"Expected context with {self.context_dim} features")
            pooled = torch.cat([pooled, context], dim=-1)
        return self.classifier(pooled).squeeze(-1)
