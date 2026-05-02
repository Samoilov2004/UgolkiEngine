"""State and action encoding for the DQN agent.

Design principles
-----------------
* **Canonical perspective** — the state is always encoded from the point of
  view of the *current player*, who is treated as "Player 1" with their pieces
  in the top-left and their goal in the bottom-right.  For Player -1, the
  board is rotated 180° and signs are flipped before encoding.  This means the
  network always sees the same topology regardless of which side it plays.

* **Flat action space** — each action is a (from_cell, to_cell) pair encoded
  as ``from_cell * 64 + to_cell`` where ``cell = row * 8 + col``.  The total
  action space is 64 × 64 = 4096.  Most actions are illegal at any given step;
  a boolean mask filters them at inference time.

* **Transform / inverse-transform** — helper functions convert real-board
  coordinates ↔ canonical (normalised) coordinates so that the DQN can always
  work in the canonical frame while the environment uses real coordinates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from corners_rl.env.moves import Move
from corners_rl.env.rules import BOARD_SIZE, PLAYER1, get_target_zone

# ── Public constants ──────────────────────────────────────────────────────────

STATE_CHANNELS: int = 3          # (my pieces, opponent pieces, my target zone)
ACTION_SPACE_SIZE: int = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 4096

# Pre-computed target-zone mask in canonical (Player-1) space: bottom-right 3×3.
_TARGET_ZONE_MASK: NDArray[np.float32] = np.zeros(
    (BOARD_SIZE, BOARD_SIZE), dtype=np.float32
)
for _r, _c in get_target_zone(PLAYER1):
    _TARGET_ZONE_MASK[_r, _c] = 1.0


# ── Coordinate transforms ─────────────────────────────────────────────────────

def transform_pos_for_player(
    pos: tuple[int, int], current_player: int
) -> tuple[int, int]:
    """Convert a real-board position to the canonical (current-player) frame.

    For Player 1 this is the identity.  For Player -1 the board is rotated
    180°, so ``(r, c) → (N-1-r, N-1-c)`` where ``N = BOARD_SIZE``.

    Args:
        pos: ``(row, col)`` in real-board coordinates.
        current_player: ``1`` or ``-1``.

    Returns:
        ``(row, col)`` in canonical coordinates.
    """
    if current_player == 1:
        return pos
    n = BOARD_SIZE - 1
    return (n - pos[0], n - pos[1])


def inverse_transform_pos_for_player(
    pos: tuple[int, int], current_player: int
) -> tuple[int, int]:
    """Convert a canonical position back to real-board coordinates.

    A 180° rotation is its own inverse, so this function is identical to
    :func:`transform_pos_for_player`.

    Args:
        pos: ``(row, col)`` in canonical coordinates.
        current_player: ``1`` or ``-1``.

    Returns:
        ``(row, col)`` in real-board coordinates.
    """
    # 180° rotation is self-inverse
    return transform_pos_for_player(pos, current_player)


def transform_move_for_player(move: Move, current_player: int) -> Move:
    """Transform every waypoint in *move* to the canonical frame.

    Args:
        move: Move in real-board coordinates (as returned by
              :func:`~corners_rl.env.moves.get_legal_moves`).
        current_player: ``1`` or ``-1``.

    Returns:
        Move with every cell transformed to canonical coordinates.
    """
    return tuple(transform_pos_for_player(cell, current_player) for cell in move)


def inverse_transform_move_for_player(move: Move, current_player: int) -> Move:
    """Convert a move from canonical coordinates back to real-board coordinates.

    Args:
        move: Move in canonical coordinates.
        current_player: ``1`` or ``-1``.

    Returns:
        Move in real-board coordinates.
    """
    return tuple(inverse_transform_pos_for_player(cell, current_player) for cell in move)


# ── State encoding ────────────────────────────────────────────────────────────

def encode_state(
    board: NDArray[np.int8], current_player: int
) -> NDArray[np.float32]:
    """Encode the board as a 3-channel float32 tensor from *current_player*'s view.

    Channels:

    * **0** — cells occupied by the current player (1.0 / 0.0).
    * **1** — cells occupied by the opponent (1.0 / 0.0).
    * **2** — cells that form the current player's target zone (1.0 / 0.0).

    The canonical frame always represents the current player as "Player 1" with
    pieces starting in the top-left corner and the goal in the bottom-right:

    * If ``current_player == 1``: board is used unchanged.
    * If ``current_player == -1``: board is rotated 180° and all values negated
      so that Player -1's pieces (originally −1) become +1 and are located in
      the top-left corner (their start zone in the canonical frame).

    Args:
        board: Raw game board, shape ``(BOARD_SIZE, BOARD_SIZE)``, dtype int8.
        current_player: ``1`` or ``-1``.

    Returns:
        Float32 array of shape ``(STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)``.
    """
    if current_player == -1:
        # Rotate 180° (flip both axes) and negate — Player -1 becomes "Player 1"
        # in the canonical frame.
        board = -board[::-1, ::-1]

    obs = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    obs[0] = (board == 1).astype(np.float32)   # my pieces
    obs[1] = (board == -1).astype(np.float32)  # opponent pieces
    obs[2] = _TARGET_ZONE_MASK                  # my target zone (always bottom-right)
    return obs


# ── Action encoding ───────────────────────────────────────────────────────────

def encode_action(move: Move) -> int:
    """Encode a move as a flat integer action ID.

    Only the *first* and *last* waypoints are used; intermediate cells in a
    jump chain are discarded.  This means multiple physical moves (e.g. two
    different chain-jump paths that share the same start and end) may map to
    the same action ID — :func:`action_id_to_move` resolves ties by choosing
    the longest path.

    Formula::

        from_cell = from_row * BOARD_SIZE + from_col
        to_cell   = to_row   * BOARD_SIZE + to_col
        action_id = from_cell * (BOARD_SIZE ** 2) + to_cell

    Args:
        move: A legal move (real or canonical coordinates).

    Returns:
        Integer in ``[0, ACTION_SPACE_SIZE)``.
    """
    from_row, from_col = move[0]
    to_row, to_col = move[-1]
    from_cell = from_row * BOARD_SIZE + from_col
    to_cell   = to_row   * BOARD_SIZE + to_col
    return from_cell * (BOARD_SIZE * BOARD_SIZE) + to_cell


def decode_action(action_id: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Decode a flat action ID back to ``(from_pos, to_pos)``.

    This is the inverse of :func:`encode_action`.

    Args:
        action_id: Integer in ``[0, ACTION_SPACE_SIZE)``.

    Returns:
        ``(from_pos, to_pos)`` where each is a ``(row, col)`` tuple.

    Raises:
        ValueError: If *action_id* is out of range.
    """
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError(
            f"action_id {action_id} is out of range [0, {ACTION_SPACE_SIZE})."
        )
    n2 = BOARD_SIZE * BOARD_SIZE  # 64
    from_cell, to_cell = divmod(action_id, n2)
    from_pos = (from_cell // BOARD_SIZE, from_cell % BOARD_SIZE)
    to_pos   = (to_cell   // BOARD_SIZE, to_cell   % BOARD_SIZE)
    return from_pos, to_pos


# ── Legal move helpers ────────────────────────────────────────────────────────

def legal_action_mask(legal_moves: list[Move]) -> NDArray[np.bool_]:
    """Build a boolean mask over the flat action space.

    The mask is ``True`` at every index that corresponds to at least one move
    in *legal_moves*.

    Args:
        legal_moves: List of legal moves in the *same* coordinate frame used
                     by the action encoding (real or canonical — must be
                     consistent with each other).

    Returns:
        Boolean array of shape ``(ACTION_SPACE_SIZE,)`` (4096 elements).
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for move in legal_moves:
        mask[encode_action(move)] = True
    return mask


def action_id_to_move(action_id: int, legal_moves: list[Move]) -> Move:
    """Map a flat action ID to a concrete legal move.

    Because :func:`encode_action` discards intermediate jump-chain waypoints,
    multiple physical moves can share the same action ID.  When this happens
    the *longest path* (most jumps) is returned — it is always at least as
    efficient as shorter alternatives.

    Args:
        action_id: Flat action ID in the same coordinate frame as *legal_moves*.
        legal_moves: The current list of legal moves.

    Returns:
        The matching :data:`~corners_rl.env.moves.Move`.

    Raises:
        ValueError: If no legal move corresponds to *action_id*.
    """
    from_pos, to_pos = decode_action(action_id)
    candidates = [
        m for m in legal_moves
        if m[0] == from_pos and m[-1] == to_pos
    ]
    if not candidates:
        raise ValueError(
            f"No legal move matches action_id {action_id} "
            f"(from={from_pos}, to={to_pos})."
        )
    # Prefer the longest chain (most jumps)
    return max(candidates, key=len)
