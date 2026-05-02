"""Neural network for the DQN value function.

Input: board observation tensor of shape ``(3, board_size, board_size)``.
Output: scalar Q-value estimate for that board state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Two-layer residual block with ReLU activations.

    Args:
        channels: Number of feature channels (constant throughout).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # TODO: define conv1, bn1, conv2, bn2
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # TODO: residual = x; x = relu(bn1(conv1(x))); x = bn2(conv2(x));
        #       return relu(x + residual)
        raise NotImplementedError


class DQNNetwork(nn.Module):
    """Convolutional network that estimates V(s) for afterstate DQN.

    Architecture:
    1. Input projection: ``in_channels → channels`` via 3×3 conv.
    2. ``num_res_blocks`` residual blocks.
    3. Global average pooling.
    4. Fully-connected head: ``channels → hidden_dim → 1``.

    Args:
        board_size: Side length of the board (used for input shape validation).
        in_channels: Number of input planes (default 3).
        channels: Convolutional feature channels (default 64).
        num_res_blocks: Number of residual blocks (default 4).
        hidden_dim: Hidden size of the FC head (default 256).
    """

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 3,
        channels: int = 64,
        num_res_blocks: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.board_size = board_size

        # TODO: define input_conv (3×3, in_channels → channels, padding=1)
        # TODO: define res_blocks as nn.Sequential of ResidualBlock(channels)
        # TODO: define global_pool as nn.AdaptiveAvgPool2d(1)
        # TODO: define fc_head: Linear(channels, hidden_dim) → ReLU → Linear(hidden_dim, 1)
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(batch, 3, board_size, board_size)``.

        Returns:
            Q-value tensor of shape ``(batch, 1)``.
        """
        # TODO: x → input_conv → res_blocks → global_pool → flatten → fc_head
        raise NotImplementedError
