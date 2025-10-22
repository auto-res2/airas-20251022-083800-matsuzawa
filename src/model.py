import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Two-hidden-layer MLP suitable for Fashion-MNIST."""

    def __init__(self, input_dim: int, hidden_units: int, output_dim: int, activation: str = "relu") -> None:
        super().__init__()
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation")

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_units),
            act,
            nn.Linear(hidden_units, hidden_units),
            act,
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, x):  # type: ignore
        return self.net(x)