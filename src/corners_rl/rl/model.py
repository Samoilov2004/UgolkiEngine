"""DQN network architecture and helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from corners_rl.rl.encoding import ACTION_SPACE_SIZE, BOARD_SIZE, STATE_CHANNELS


# ── Network ───────────────────────────────────────────────────────────────────

class DQNModel(nn.Module):
    """Convolutional Q-network for the Corners game.

    Maps a board observation to Q-values for every possible (from, to) action.

    Architecture::

        Conv2d(3, 32, 3, padding=1) → ReLU
        Conv2d(32, 64, 3, padding=1) → ReLU
        Flatten
        Linear(64*8*8, 512) → ReLU
        Linear(512, 4096)

    Args:
        in_channels: Number of input planes (default ``STATE_CHANNELS`` = 3).
        board_size: Side length of the board (default ``BOARD_SIZE`` = 8).
        action_size: Output size (default ``ACTION_SPACE_SIZE`` = 4096).
    """

    def __init__(
        self,
        in_channels: int = STATE_CHANNELS,
        board_size: int = BOARD_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
    ) -> None:
        super().__init__()
        flat_size = 64 * board_size * board_size  # 64 * 8 * 8 = 4096

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Float32 tensor of shape ``(batch, in_channels, board_size, board_size)``.

        Returns:
            Q-value tensor of shape ``(batch, action_size)``.
        """
        return self.fc(self.conv(x))


# ── Masked argmax ─────────────────────────────────────────────────────────────

def masked_argmax(
    q_values: Tensor,
    legal_mask: NDArray[np.bool_],
) -> int:
    """Return the index of the highest Q-value among legal actions.

    All illegal actions (``legal_mask[i] == False``) are set to ``-∞`` before
    taking the argmax, so they can never be selected.

    Args:
        q_values: 1-D tensor of shape ``(ACTION_SPACE_SIZE,)``.
        legal_mask: Boolean array of shape ``(ACTION_SPACE_SIZE,)``.
                    ``True`` means the action is legal.

    Returns:
        Integer action ID (index into *q_values*).

    Raises:
        ValueError: If *legal_mask* contains no ``True`` entries.
    """
    if not legal_mask.any():
        raise ValueError("legal_mask has no True entries — no legal actions available.")

    mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=q_values.device)
    q_masked = q_values.clone()
    q_masked[~mask_t] = float("-inf")
    return int(q_masked.argmax().item())
