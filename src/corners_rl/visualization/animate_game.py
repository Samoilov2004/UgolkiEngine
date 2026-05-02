"""Game recording and animated GIF/MP4 export."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests

from corners_rl.agents.base import BaseAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.visualization.board_plot import plot_board

# ── Public types ───────────────────────────────────────────────────────────────

Frame = dict  # keys: board, move, current_player, winner, move_number


# ── Recording ─────────────────────────────────────────────────────────────────

def record_game(
    agent1: BaseAgent,
    agent2: BaseAgent,
    max_moves: int = 300,
    seed: Optional[int] = None,
) -> list[Frame]:
    """Play a game and record every board state as a list of frame dicts.

    Each frame captures the state *after* the move was applied, together with
    metadata that is used when rendering the animated GIF.

    Args:
        agent1: Agent that controls Player 1 (moves first).
        agent2: Agent that controls Player −1.
        max_moves: Episode step limit forwarded to :class:`CornersEnv`.
        seed: When given, seeds agents' internal ``_rng`` attributes for
              reproducible replays.

    Returns:
        List of frame dicts, each with keys:

        * ``board``          — ``(8, 8)`` int8 array *after* the move.
        * ``move``           — the move that was just played.
        * ``current_player`` — the player who made this move.
        * ``winner``         — ``1``, ``-1``, or ``None`` (only set on last frame).
        * ``move_number``    — 1-indexed half-move counter.
    """
    import random as _random

    env = CornersEnv(max_moves=max_moves)
    env.reset()

    if seed is not None:
        master_rng = _random.Random(seed)
        for agent in (agent1, agent2):
            if hasattr(agent, "_rng") and isinstance(agent._rng, _random.Random):
                agent._rng = _random.Random(master_rng.randint(0, 2**32 - 1))

    agent_map: dict[int, BaseAgent] = {1: agent1, -1: agent2}
    frames: list[Frame] = []

    # Record the initial state as frame 0
    frames.append(
        {
            "board": env.board.copy(),
            "move": None,
            "current_player": env.current_player,
            "winner": None,
            "move_number": 0,
        }
    )

    while not env.is_terminal():
        move = agent_map[env.current_player].select_move(env)
        _, _, done, info = env.step(move)

        # Use info fields: move and player_moved are authoritative (real coords).
        frames.append(
            {
                "board":          info["board_after"],       # board after the move
                "move":           info["move"],              # real-board coordinates
                "current_player": info["player_moved"],      # who just moved
                "winner":         info["winner"] if done else None,
                "move_number":    env.move_count,
            }
        )

    return frames


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _frame_to_rgb(frame: Frame, agent1_name: str, agent2_name: str) -> np.ndarray:
    """Render one frame as an RGB uint8 array."""
    board  = frame["board"]
    move   = frame["move"]
    mnum   = frame["move_number"]
    player = frame["current_player"]
    winner = frame["winner"]

    if mnum == 0:
        title = f"Start — {agent1_name} (P1, blue) vs {agent2_name} (P−1, red)"
    elif winner is not None:
        if winner == 1:
            result_str = f"{agent1_name} wins!"
        elif winner == -1:
            result_str = f"{agent2_name} wins!"
        else:
            result_str = "Draw"
        title = f"Move {mnum} — {result_str}"
    else:
        player_label = agent1_name if player == 1 else agent2_name
        title = f"Move {mnum} — {player_label} (P{'1' if player == 1 else '−1'}) just moved"

    fig, _ = plot_board(board, title=title, last_move=move, target_zones=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    rgb = imageio.imread(buf)
    buf.close()
    return rgb


# ── GIF export ────────────────────────────────────────────────────────────────

def save_game_gif(
    frames: list[Frame],
    path: str | Path,
    fps: float = 2.0,
    agent1_name: str = "agent1",
    agent2_name: str = "agent2",
    hold_last: int = 3,
) -> None:
    """Save a list of game frames as an animated GIF.

    Args:
        frames: As returned by :func:`record_game`.
        path: Destination ``.gif`` file path.
        fps: Frames per second.
        agent1_name: Label shown in titles for Player 1.
        agent2_name: Label shown in titles for Player −1.
        hold_last: Number of times the final frame is repeated so the result
                   is visible before the GIF loops.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for i, frame in enumerate(frames):
        rgb = _frame_to_rgb(frame, agent1_name, agent2_name)
        # Hold the last frame longer
        repeat = hold_last if i == len(frames) - 1 else 1
        images.extend([rgb] * repeat)

    duration_ms = int(1000 / fps)
    imageio.mimsave(str(path), images, format="GIF", duration=duration_ms, loop=0)


# ── MP4 export (optional) ─────────────────────────────────────────────────────

def save_game_mp4(
    frames: list[Frame],
    path: str | Path,
    fps: float = 2.0,
    agent1_name: str = "agent1",
    agent2_name: str = "agent2",
    hold_last: int = 3,
) -> None:
    """Save a list of game frames as an MP4 video.

    Requires ``imageio[ffmpeg]`` (``pip install imageio[ffmpeg]``).  Raises
    :class:`RuntimeError` with an actionable message if ffmpeg is unavailable.

    Args:
        frames: As returned by :func:`record_game`.
        path: Destination ``.mp4`` file path.
        fps: Frames per second.
        agent1_name: Label for Player 1.
        agent2_name: Label for Player −1.
        hold_last: Repeat count for the final frame.
    """
    try:
        import imageio.v2 as _iio
        writer = _iio.get_writer  # probe availability
    except Exception as exc:
        raise RuntimeError(
            "MP4 export requires ffmpeg.  Install it with:\n"
            "  pip install imageio[ffmpeg]\n"
            "or\n"
            "  conda install ffmpeg"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for i, frame in enumerate(frames):
        rgb = _frame_to_rgb(frame, agent1_name, agent2_name)
        repeat = hold_last if i == len(frames) - 1 else 1
        images.extend([rgb] * repeat)

    with imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for img in images:
            writer.append_data(img)
