"""Static board visualisation using matplotlib."""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from corners_rl.env.rules import PLAYER1, PLAYER2, get_start_zone, get_target_zone

# ── Colour palette ─────────────────────────────────────────────────────────────

_P1_PIECE   = "#4A90D9"   # cornflower blue
_P2_PIECE   = "#E05252"   # soft red
_P1_ZONE    = "#4A90D9"   # same hue, low alpha fill
_P2_ZONE    = "#E05252"
_CELL_LIGHT = "#F5F5DC"   # beige
_CELL_DARK  = "#C8B98A"   # tan
_GRID_COLOR = "#888888"
_ARROW_COLOR = "#FFD700"   # gold — visible on both piece colours


def plot_board(
    board: NDArray[np.int8],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    last_move: Optional[tuple] = None,
    target_zones: bool = True,
) -> tuple[Figure, Axes]:
    """Draw the 8×8 Corners board as a chess-style grid.

    Args:
        board: Integer board array, shape ``(8, 8)``, values in ``{-1, 0, 1}``.
        ax: Existing :class:`matplotlib.axes.Axes` to draw into.  A new figure
            is created when ``None``.
        title: Optional title string shown above the board.
        last_move: Move tuple whose first element is ``(from_row, from_col)``
            and last element is ``(to_row, to_col)``.  When provided, a gold
            arrow is drawn from start to end.
        target_zones: When ``True``, shade the target zones of each player with
            a semi-transparent fill.

    Returns:
        ``(fig, ax)`` — the matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure  # type: ignore[assignment]

    n = 8

    # ── Checkerboard background ───────────────────────────────────────────────
    for r in range(n):
        for c in range(n):
            color = _CELL_LIGHT if (r + c) % 2 == 0 else _CELL_DARK
            ax.add_patch(mpatches.Rectangle((c, r), 1, 1, color=color, zorder=0))

    # ── Zone shading ──────────────────────────────────────────────────────────
    if target_zones:
        for player, face_color in ((PLAYER1, _P1_ZONE), (PLAYER2, _P2_ZONE)):
            for zone_cells, alpha in (
                (get_start_zone(player),  0.10),
                (get_target_zone(player), 0.25),
            ):
                for r, c in zone_cells:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (c, r), 1, 1,
                            facecolor=face_color,
                            alpha=alpha,
                            zorder=1,
                        )
                    )

    # ── Grid lines ────────────────────────────────────────────────────────────
    for i in range(n + 1):
        ax.axhline(i, color=_GRID_COLOR, linewidth=0.8, zorder=2)
        ax.axvline(i, color=_GRID_COLOR, linewidth=0.8, zorder=2)

    # ── Pieces ────────────────────────────────────────────────────────────────
    for r in range(n):
        for c in range(n):
            val = int(board[r, c])
            if val == 0:
                continue
            color = _P1_PIECE if val == PLAYER1 else _P2_PIECE
            circle = mpatches.Circle(
                (c + 0.5, r + 0.5),
                radius=0.38,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(circle)
            # Small marker in the piece centre for visual pop
            ax.plot(c + 0.5, r + 0.5, "o", color="white", markersize=3, zorder=4)

    # ── Last-move polyline ────────────────────────────────────────────────────
    # Draw one arrow segment per hop so chain jumps appear as a connected path
    # rather than a single "diagonal-looking" line from start to end.
    if last_move is not None and len(last_move) >= 2:
        for i in range(len(last_move) - 1):
            ar, ac = last_move[i]
            br, bc = last_move[i + 1]
            is_last_seg = (i == len(last_move) - 2)
            ax.annotate(
                "",
                xy=(bc + 0.5, br + 0.5),
                xytext=(ac + 0.5, ar + 0.5),
                arrowprops=dict(
                    arrowstyle="-|>" if is_last_seg else "-",
                    color=_ARROW_COLOR,
                    lw=2.5,
                    mutation_scale=20,
                ),
                zorder=5,
            )

    # ── Axes cosmetics ────────────────────────────────────────────────────────
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")

    # Row labels (0–7 from bottom) → display reversed so row 0 is at the top
    ax.set_xticks([c + 0.5 for c in range(n)])
    ax.set_xticklabels([str(c) for c in range(n)], fontsize=9)
    ax.set_yticks([r + 0.5 for r in range(n)])
    ax.set_yticklabels([str(n - 1 - r) for r in range(n)], fontsize=9)
    ax.tick_params(length=0)

    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_P1_PIECE, edgecolor="white", label="Player 1"),
        mpatches.Patch(facecolor=_P2_PIECE, edgecolor="white", label="Player −1"),
        mpatches.Patch(facecolor=_P1_ZONE,  alpha=0.35, label="P1 target zone"),
        mpatches.Patch(facecolor=_P2_ZONE,  alpha=0.35, label="P−1 target zone"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        handlelength=1.2,
    )

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()
    return fig, ax
