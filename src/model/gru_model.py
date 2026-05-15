"""
GRU Language Model - Model Module

Author: Apala Pramanik
Description: Multi-layer GRU language model with embedding and output projection layers.
"""

import torch.nn as nn


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
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.out_dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x : (B, T) - input token indices
        returns : (B, T, vocab_size) - logits for next token prediction
        """
        x = self.emb_dropout(self.embed(x))
        out, _ = self.gru(x)
        return self.output_head(self.out_dropout(out))
