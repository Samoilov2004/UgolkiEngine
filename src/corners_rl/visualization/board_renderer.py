"""Board renderer: ASCII, matplotlib static image, and animated GIF."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from corners_rl.env.board import Board, PLAYER1, PLAYER2, EMPTY


class BoardRenderer:
    """Renders :class:`~corners_rl.env.board.Board` states.

    Supports three output modes:
    * **ansi** — coloured ASCII string for terminal display.
    * **rgb_array** — NumPy uint8 ``(H, W, 3)`` image (via matplotlib).
    * **gif** — animated GIF from a sequence of board states (via imageio).

    Args:
        cell_size: Pixel size of each board cell in image renders (default 60).
        colors: Dict mapping piece values to RGB tuples.
    """

    DEFAULT_COLORS: dict[int, tuple[int, int, int]] = {
        PLAYER1: (70, 130, 180),   # steel blue
        PLAYER2: (220, 80, 80),    # soft red
        EMPTY:   (245, 245, 220),  # beige
    }

    def __init__(
        self,
        cell_size: int = 60,
        colors: Optional[dict[int, tuple[int, int, int]]] = None,
    ) -> None:
        self.cell_size = cell_size
        self.colors = colors or self.DEFAULT_COLORS

    # ------------------------------------------------------------------ #
    #  Render modes                                                        #
    # ------------------------------------------------------------------ #

    def render_ansi(self, board: Board) -> str:
        """Return a coloured ASCII representation of the board.

        Goal zones are highlighted.  Current turn information is not included
        here; the caller should prepend it.
        """
        # TODO: iterate board.grid, map values to ANSI-coloured symbols,
        #       highlight goal zone cells with background colour.
        raise NotImplementedError

    def render_rgb(self, board: Board) -> NDArray[np.uint8]:
        """Render the board as an RGB pixel array.

        Returns:
            uint8 array of shape ``(H, W, 3)`` where
            ``H = W = board.board_size * cell_size``.
        """
        # TODO: create canvas, fill cells, draw grid lines, draw pieces,
        #       shade goal zones with a translucent overlay.
        raise NotImplementedError

    def save_gif(
        self,
        frames: Sequence[Board],
        output_path: Path | str,
        fps: int = 4,
    ) -> None:
        """Save a sequence of board states as an animated GIF.

        Args:
            frames: Ordered list of board states to animate.
            output_path: Destination path for the GIF file.
            fps: Frames per second.
        """
        # TODO: render each frame to RGB via render_rgb(), collect list of arrays,
        #       call imageio.mimsave(output_path, arrays, fps=fps).
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _draw_piece(
        self,
        canvas: NDArray[np.uint8],
        row: int,
        col: int,
        player: int,
    ) -> None:
        """Draw a circular piece on ``canvas`` at ``(row, col)``."""
        # TODO: use a filled circle drawn with matplotlib or direct array ops.
        raise NotImplementedError
