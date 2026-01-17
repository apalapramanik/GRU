"""
GRU Language Model - Model Module

Author: Apala Pramanik
Description: Multi-layer GRU language model with embedding and output projection layers.
"""

import torch
import torch.nn as nn
from .gru_cell import GRUCell


class GRULanguageModel(nn.Module):
    """
    Character-level GRU language model.

    Architecture:
    tokens → embedding → GRU → linear head
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.layers = nn.ModuleList(
            [
                GRUCell(
                    embed_dim if i == 0 else hidden_dim,
                    hidden_dim
                )
                for i in range(num_layers)
            ]
        )

        self.output_head = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        """
        x : (B, T) - input token indices
        returns : (B, T, vocab_size) - logits for next token prediction
        """

        B, T = x.size()
        x = self.embed(x)

        # Initialize hidden states
        h = [
            torch.zeros(B, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        for t in range(T):
            inp = x[:, t]

            for layer_idx, layer in enumerate(self.layers):
                h[layer_idx] = layer(inp, h[layer_idx])
                inp = h[layer_idx]

            outputs.append(inp.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        logits = self.output_head(outputs)

        return logits
