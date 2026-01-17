import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """
    Single GRU cell implemented from scratch.

    Gates:
    - update gate (z)
    - reset gate (r)
    - candidate hidden state (h_tilde)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gates
        self.linear_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        """
        x : (B, input_dim)
        h : (B, hidden_dim)
        """

        combined = torch.cat([x, h], dim=1)

        z = torch.sigmoid(self.linear_z(combined))  # update gate
        r = torch.sigmoid(self.linear_r(combined))  # reset gate

        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.linear_h(combined_reset))

        h_next = (1 - z) * h + z * h_tilde

        return h_next
