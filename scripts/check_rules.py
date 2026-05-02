#!/usr/bin/env python3
"""Debug script: verify that every move in every game obeys the orthogonal rules.

Usage
-----
    python scripts/check_rules.py
    python scripts/check_rules.py --games 50 --max-moves 300 --seed 42

The script checks:
1. Every legal move in the initial position passes :func:`validate_move`.
2. No legal move in the initial position contains a diagonal segment.
3. For *--games* games of Random vs Random, every applied move:
   - passes :func:`validate_move` on the board *before* the move;
   - contains no diagonal segment (all steps are orthogonal);
   - for jumps: the midpoint cell is occupied on the board before the move;
   - for jumps: the midpoint cell is occupied by some piece (not empty).
4. Prints a short summary and exits with code 1 if any violation is found.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.agents.random_agent import RandomAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import (
    get_legal_moves,
    is_orthogonal_jump,
    is_orthogonal_step,
    midpoint,
    validate_move,
)
from corners_rl.env.rules import EMPTY, PLAYER1, PLAYER2, initial_board


def _check_move_orthogonal(
    board_before,
    move,
    player: int,
    context: str,
) -> list[str]:
    """Return a list of violation strings (empty = no violations)."""
    violations: list[str] = []

    try:
        validate_move(board_before, move, player)
    except ValueError as exc:
        violations.append(f"[{context}] validate_move FAILED: {exc}")
        return violations  # no point checking geometry if basic validation fails

    for i in range(len(move) - 1):
        a, b = move[i], move[i + 1]
        dr = b[0] - a[0]
        dc = b[1] - a[1]

        # Diagonal segment?
        if abs(dr) > 0 and abs(dc) > 0:
            violations.append(
                f"[{context}] Diagonal segment {a}→{b} "
                f"(Δrow={dr}, Δcol={dc}) in move {move}"
            )
            continue

        if is_orthogonal_jump(a, b):
            mid = midpoint(a, b)
            if board_before[mid[0], mid[1]] == EMPTY:
                violations.append(
                    f"[{context}] Jump {a}→{b} over empty midpoint {mid} "
                    f"in move {move}"
                )
        elif not is_orthogonal_step(a, b):
            violations.append(
                f"[{context}] Unexpected segment {a}→{b} "
                f"(Δrow={dr}, Δcol={dc}) in move {move}"
            )

    return violations


def check_initial_position() -> list[str]:
    """Check all legal moves from the standard initial board."""
    board = initial_board()
    violations: list[str] = []
    for player in (PLAYER1, PLAYER2):
        for move in get_legal_moves(board, player):
            violations.extend(
                _check_move_orthogonal(board, move, player, f"initial P{player}")
            )
    return violations


def check_games(n_games: int, max_moves: int, seed: int) -> list[str]:
    """Play *n_games* Random-vs-Random games and validate every applied move."""
    rng = random.Random(seed)
    violations: list[str] = []
    total_moves = 0

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**32 - 1)
        env = CornersEnv(max_moves=max_moves)
        env.reset()
        a1 = RandomAgent(seed=game_seed)
        a2 = RandomAgent(seed=game_seed + 1)
        agent_map = {1: a1, -1: a2}

        while not env.is_terminal():
            board_before = env.board
            player = env.current_player
            move = agent_map[player].select_move(env)

            _, _, _, info = env.step(move)
            applied_move = info["move"]
            player_moved = info["player_moved"]
            total_moves += 1

            ctx = f"game {game_idx + 1}, move {env.move_count}, P{player_moved}"
            vs = _check_move_orthogonal(board_before, applied_move, player_moved, ctx)
            violations.extend(vs)

    print(
        f"Checked {n_games} games, {total_moves} moves total.",
        flush=True,
    )
    return violations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify that all game moves obey the orthogonal rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games",     type=int, default=50,  help="Number of games.")
    p.add_argument("--max-moves", type=int, default=300, dest="max_moves",
                   help="Move limit per game.")
    p.add_argument("--seed",      type=int, default=42,  help="Random seed.")
    p.add_argument("--quiet",     action="store_true",
                   help="Only print summary, not individual violations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Corners rules checker ===")
    print(f"Config: games={args.games}, max_moves={args.max_moves}, seed={args.seed}")
    print()

    # 1. Initial position
    print("Checking initial-position legal moves …", end=" ", flush=True)
    v_init = check_initial_position()
    status = "OK" if not v_init else f"FAIL ({len(v_init)} violations)"
    print(status)

    # 2. Random games
    print(f"Checking {args.games} Random-vs-Random games …")
    v_games = check_games(args.games, args.max_moves, args.seed)
    status = "OK" if not v_games else f"FAIL ({len(v_games)} violations)"
    print(f"Game check: {status}")

    all_violations = v_init + v_games

    if all_violations and not args.quiet:
        print()
        print("=== Violations ===")
        for v in all_violations[:50]:   # cap output
            print(" •", v)
        if len(all_violations) > 50:
            print(f"  … and {len(all_violations) - 50} more.")

    print()
    if all_violations:
        print(f"RESULT: FAILED — {len(all_violations)} violation(s) found.")
        sys.exit(1)
    else:
        print("RESULT: PASSED — all moves are orthogonal and valid.")


if __name__ == "__main__":
    main()
