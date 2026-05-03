"""DQN agent: epsilon-greedy policy over masked Q-values."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch

from corners_rl.agents.base import BaseAgent
from corners_rl.env.moves import filter_forward_moves
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    action_id_to_move,
    encode_state,
    inverse_transform_move_for_player,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel, masked_argmax

if TYPE_CHECKING:
    from corners_rl.env.corners_env import CornersEnv
    from corners_rl.env.moves import Move


class DQNAgent(BaseAgent):
    """Epsilon-greedy DQN agent.

    Uses a :class:`~corners_rl.rl.model.DQNModel` to estimate Q-values and
    applies an action mask so that only legal moves can be selected.

    All state/action encoding uses the canonical (current-player) frame — the
    network always perceives itself as "Player 1" regardless of which side it
    actually controls.

    Args:
        model: The Q-network.  If ``None``, a freshly initialised
               :class:`~corners_rl.rl.model.DQNModel` is created.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        epsilon: Initial exploration probability for ε-greedy.
        name: Agent label.
        seed: Random seed for reproducible exploration.
    """

    def __init__(
        self,
        model: Optional[DQNModel] = None,
        device: Union[str, torch.device] = "cpu",
        epsilon: float = 1.0,
        name: str = "dqn",
        seed: Optional[int] = None,
        forward_only: bool = False,
    ) -> None:
        super().__init__(name)
        self._device = torch.device(device)
        self._model  = (model or DQNModel()).to(self._device)
        self._epsilon = float(epsilon)
        self._rng = random.Random(seed)
        self._forward_only = forward_only

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def select_move(self, env: "CornersEnv") -> "Move":
        """Choose a move using ε-greedy with a legal-action mask.

        Steps:

        1. Collect ``env.legal_moves()`` (real-board coordinates).
        2. Transform every move to the canonical frame.
        3. Build a legal-action mask over the 4096-action space.
        4. Encode the board state as a (3, 8, 8) tensor.
        5. With probability ``epsilon`` pick a uniformly random legal move;
           otherwise forward through the model and pick the legal argmax.
        6. Inverse-transform the chosen canonical move back to real-board
           coordinates and return it.

        Args:
            env: The live game environment (read-only).

        Returns:
            A legal :data:`~corners_rl.env.moves.Move` in real-board coordinates.

        Raises:
            ValueError: If there are no legal moves (degenerate state).
        """
        player = env.current_player
        real_moves = env.legal_moves()

        if not real_moves:
            raise ValueError(
                f"No legal moves for player {player}. "
                "Board may be in a degenerate state."
            )

        if self._forward_only:
            real_moves = filter_forward_moves(real_moves, player)

        # ── canonical frame ───────────────────────────────────────────────
        canonical_moves = [
            transform_move_for_player(m, player) for m in real_moves
        ]

        if self._rng.random() < self._epsilon:
            # Exploration: uniformly random legal canonical move
            chosen_canonical = self._rng.choice(canonical_moves)
        else:
            # Exploitation: masked argmax over Q-values
            state_arr = encode_state(env.board, player)          # (3,8,8) float32
            state_t   = (
                torch.from_numpy(state_arr)
                .unsqueeze(0)          # (1, 3, 8, 8)
                .to(self._device)
            )
            mask = legal_action_mask(canonical_moves)            # (4096,) bool

            self._model.eval()
            with torch.no_grad():
                q_values = self._model(state_t).squeeze(0)      # (4096,)

            action_id     = masked_argmax(q_values, mask)
            chosen_canonical = action_id_to_move(action_id, canonical_moves)

        # ── back to real coordinates ──────────────────────────────────────
        return inverse_transform_move_for_player(chosen_canonical, player)

    # ── Epsilon control ───────────────────────────────────────────────────────

    def set_epsilon(self, epsilon: float) -> None:
        """Set the exploration probability.

        Args:
            epsilon: New ε value, clamped to ``[0.0, 1.0]``.
        """
        self._epsilon = float(max(0.0, min(1.0, epsilon)))

    @property
    def epsilon(self) -> float:
        """Current exploration probability."""
        return self._epsilon

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[Path, str]) -> None:
        """Save the model weights and agent metadata to a checkpoint file.

        Args:
            path: Destination file path (e.g. ``"checkpoints/dqn_ep1000.pt"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "epsilon": self._epsilon,
                "name": self._name,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        device: Union[str, torch.device] = "cpu",
        epsilon: Optional[float] = None,
        forward_only: bool = False,
    ) -> "DQNAgent":
        """Load a :class:`DQNAgent` from a checkpoint file.

        Args:
            path: Path to a checkpoint saved with :meth:`save`.
            device: Device on which to load the model.
            epsilon: Override the saved epsilon (useful to force greedy play).
            forward_only: Restrict moves to those going toward the target zone.

        Returns:
            A fully initialised :class:`DQNAgent`.
        """
        from corners_rl.utils.seeding import resolve_device
        resolved = str(resolve_device(device))
        ckpt = torch.load(path, map_location=resolved, weights_only=True)
        model = DQNModel()
        model.load_state_dict(ckpt["model_state_dict"])
        saved_epsilon = ckpt.get("epsilon", 0.0)
        return cls(
            model=model,
            device=resolved,
            epsilon=epsilon if epsilon is not None else saved_epsilon,
            name=ckpt.get("name", "dqn"),
            forward_only=forward_only,
        )

    def __repr__(self) -> str:
        return (
            f"DQNAgent(name={self._name!r}, "
            f"epsilon={self._epsilon:.3f}, "
            f"device={self._device})"
        )
