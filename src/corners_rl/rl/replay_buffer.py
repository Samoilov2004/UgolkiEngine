"""Replay buffers for DQN experience replay.

Two implementations are provided:

* :class:`ReplayBuffer` — standard uniform random sampling.
* :class:`PrioritizedReplayBuffer` — Prioritized Experience Replay (PER),
  Schaul et al. 2016 (https://arxiv.org/abs/1511.05952).

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


# ── Prioritized Experience Replay ─────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer (Schaul et al., 2016).

    Transitions are sampled with probability proportional to their priority::

        priority_i  = (|td_error_i| + priority_epsilon) ** alpha
        P(i)        = priority_i / sum(priority_j for j in buffer)

    Importance-sampling weights correct for the non-uniform sampling::

        w_i = (N * P(i)) ** (-beta)    # then normalised by max(w)

    ``beta`` should be annealed from a small initial value (e.g. 0.4) to 1.0
    over training; the caller is responsible for this schedule.

    .. note::
        This implementation uses :func:`numpy.random.Generator.choice` with an
        explicit probability array — O(N) time and memory per call.  This is
        appropriate for a research prototype (capacity ≤ 100 000).  For
        production-scale buffers, a sum-tree structure reduces sampling to
        O(log N).

    Args:
        capacity: Maximum number of transitions to retain.
        alpha: Prioritisation exponent.  ``alpha=0`` recovers uniform sampling;
               ``alpha=1`` samples fully proportional to TD error.
        priority_epsilon: Small constant added to every TD error before raising
                          to ``alpha``, ensuring no transition has zero priority.
        seed: Optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self._capacity = capacity
        self._alpha = alpha
        self._priority_epsilon = priority_epsilon
        self._rng = np.random.default_rng(seed)
        self._pos: int = 0
        self._size: int = 0

        # Transition storage (identical layout to ReplayBuffer)
        self._states      = np.empty((capacity, *_STATE_SHAPE), dtype=np.float32)
        self._actions     = np.empty(capacity, dtype=np.int64)
        self._rewards     = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, *_STATE_SHAPE), dtype=np.float32)
        self._dones       = np.empty(capacity, dtype=np.bool_)
        self._next_masks_packed = np.empty((capacity, _MASK_PACKED_LEN), dtype=np.uint8)

        # Priority storage — each slot holds (|δ| + ε)^α
        self._priorities  = np.zeros(capacity, dtype=np.float64)
        self._max_priority: float = 1.0   # new transitions receive this priority

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
        """Add one transition, assigning it the current maximum priority.

        New transitions always receive ``max_priority`` so they are guaranteed
        to be sampled at least once before their TD error is known.

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
        self._priorities[p]  = self._max_priority

        self._pos  = (p + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> dict:
        """Sample a prioritised mini-batch with importance-sampling weights.

        Args:
            batch_size: Number of transitions to sample (with replacement).
            beta: IS-weight exponent in ``[0, 1]``.  ``beta=0`` disables
                  correction; ``beta=1`` fully corrects for non-uniform
                  sampling.

        Returns:
            Dict with torch.Tensor values:

            * ``"states"``           — float32, shape ``(B, 3, 8, 8)``
            * ``"actions"``          — int64,   shape ``(B,)``
            * ``"rewards"``          — float32, shape ``(B,)``
            * ``"next_states"``      — float32, shape ``(B, 3, 8, 8)``
            * ``"dones"``            — bool,    shape ``(B,)``
            * ``"next_legal_masks"`` — bool,    shape ``(B, 4096)``
            * ``"indices"``          — ndarray int64, shape ``(B,)``  — pass to
                                       :meth:`update_priorities` after the
                                       gradient step.
            * ``"weights"``          — float32 tensor, shape ``(B,)`` — multiply
                                       element-wise with the TD loss before
                                       backprop.

        Raises:
            ValueError: If *batch_size* exceeds the current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only contains "
                f"{self._size} transitions."
            )

        # Sampling probabilities P(i) = priority_i / Σ priority_j
        priorities = self._priorities[:self._size]
        prob_sum   = priorities.sum()
        probs      = priorities / prob_sum               # (N,)

        idx = self._rng.choice(self._size, size=batch_size, replace=True, p=probs)

        # Importance-sampling weights: w_i = (N · P(i))^(−β), normalised
        raw_weights = (self._size * probs[idx]) ** (-beta)  # (B,)
        weights     = (raw_weights / raw_weights.max()).astype(np.float32)

        # Unpack bit masks → (B, 4096) bool
        packed = self._next_masks_packed[idx]
        masks  = np.unpackbits(packed, axis=1)[:, :ACTION_SPACE_SIZE].astype(np.bool_)

        return {
            "states":           torch.from_numpy(self._states[idx]),
            "actions":          torch.from_numpy(self._actions[idx]),
            "rewards":          torch.from_numpy(self._rewards[idx]),
            "next_states":      torch.from_numpy(self._next_states[idx]),
            "dones":            torch.from_numpy(self._dones[idx]),
            "next_legal_masks": torch.from_numpy(masks),
            "indices":          idx,
            "weights":          torch.from_numpy(weights),
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update the stored priorities for a batch of transitions.

        Should be called immediately after the gradient step, using the
        ``"indices"`` returned by :meth:`sample` and the per-sample TD errors
        computed during the update.

        Args:
            indices: Buffer indices as returned by :meth:`sample`
                     (``batch["indices"]``).
            td_errors: Per-sample TD errors, shape ``(B,)``.  May be a numpy
                       array or a torch Tensor (converted automatically).
        """
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        new_priorities = (np.abs(td_errors) + self._priority_epsilon) ** self._alpha
        self._priorities[indices] = new_priorities
        candidate = float(new_priorities.max())
        if candidate > self._max_priority:
            self._max_priority = candidate

    def stats(self) -> dict[str, float]:
        """Return descriptive statistics of the current priority distribution.

        Returns:
            Dict with keys ``priority_mean``, ``priority_max``,
            ``priority_min``, ``priority_std``.
        """
        p = self._priorities[:self._size]
        return {
            "priority_mean": float(p.mean()),
            "priority_max":  float(p.max()),
            "priority_min":  float(p.min()),
            "priority_std":  float(p.std()),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def alpha(self) -> float:
        return self._alpha

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer("
            f"size={self._size}/{self._capacity}, "
            f"alpha={self._alpha})"
        )
