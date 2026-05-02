"""Circular replay buffer for DQN experience replay.

Memory layout
-------------
All fields are stored in pre-allocated NumPy arrays for efficient
random-access sampling.  The ``next_legal_mask`` field is stored in
bit-packed form (``np.packbits``) to keep memory consumption low:
512 bytes per transition instead of 4096 bytes.

Memory estimate at capacity ``C``::

    states / next_states : C × 3 × 8 × 8 × 4 B  ≈ C × 768 B
    next_legal_masks      : C × 512 B   (bit-packed)
    actions / rewards / dones : negligible

At C = 100 000 the total is ≈ 200 MB.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from corners_rl.rl.encoding import ACTION_SPACE_SIZE, BOARD_SIZE, STATE_CHANNELS

# Shape constants
_STATE_SHAPE = (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)   # (3, 8, 8)
_MASK_PACKED_LEN = ACTION_SPACE_SIZE // 8                  # 512


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Stores ``(state, action, reward, next_state, done, next_legal_mask)``
    transitions and supports uniform random sampling.

    Args:
        capacity: Maximum number of transitions to retain (oldest are
                  overwritten when the buffer is full).
        seed: Optional random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._pos: int = 0
        self._size: int = 0

        # Pre-allocated storage
        self._states      = np.empty((capacity, *_STATE_SHAPE), dtype=np.float32)
        self._actions     = np.empty(capacity, dtype=np.int64)
        self._rewards     = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, *_STATE_SHAPE), dtype=np.float32)
        self._dones       = np.empty(capacity, dtype=np.bool_)
        # Bit-packed legal masks — 512 uint8 per transition
        self._next_masks_packed = np.empty((capacity, _MASK_PACKED_LEN), dtype=np.uint8)

    # ── Public API ────────────────────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        """Add one transition to the buffer (overwrites oldest if full).

        Args:
            state: Board encoding, shape ``(3, 8, 8)``, dtype float32.
            action: Flat action ID (canonical frame).
            reward: Scalar reward received.
            next_state: Board encoding after the move, shape ``(3, 8, 8)``.
            done: Whether the episode ended.
            next_legal_mask: Boolean mask of legal actions in ``next_state``,
                             shape ``(4096,)``.
        """
        p = self._pos
        self._states[p]      = state
        self._actions[p]     = action
        self._rewards[p]     = reward
        self._next_states[p] = next_state
        self._dones[p]       = done
        self._next_masks_packed[p] = np.packbits(next_legal_mask)

        self._pos  = (p + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random mini-batch.

        Args:
            batch_size: Number of transitions to sample (without replacement).

        Returns:
            Dict with torch.Tensor values:

            * ``"states"``          — float32, shape ``(B, 3, 8, 8)``
            * ``"actions"``         — int64,   shape ``(B,)``
            * ``"rewards"``         — float32, shape ``(B,)``
            * ``"next_states"``     — float32, shape ``(B, 3, 8, 8)``
            * ``"dones"``           — bool,    shape ``(B,)``
            * ``"next_legal_masks"``— bool,    shape ``(B, 4096)``

        Raises:
            ValueError: If *batch_size* exceeds the current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only contains "
                f"{self._size} transitions."
            )
        idx = self._rng.choice(self._size, size=batch_size, replace=False)

        # Unpack bit masks → (B, 4096) bool
        packed = self._next_masks_packed[idx]             # (B, 512)
        masks  = np.unpackbits(packed, axis=1)[:, :ACTION_SPACE_SIZE].astype(np.bool_)

        return {
            "states":           torch.from_numpy(self._states[idx]),
            "actions":          torch.from_numpy(self._actions[idx]),
            "rewards":          torch.from_numpy(self._rewards[idx]),
            "next_states":      torch.from_numpy(self._next_states[idx]),
            "dones":            torch.from_numpy(self._dones[idx]),
            "next_legal_masks": torch.from_numpy(masks),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={self._size}/{self._capacity})"
