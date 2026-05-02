#!/usr/bin/env python3
"""Comprehensive diagnostic script for the DQN training pipeline.

Checks 7 areas: checkpoint, action sanity, evaluation sanity, reward sanity,
training step sanity, perspective/transform sanity, and summary calculation.

Usage
-----
    python scripts/debug_dqn_training.py --checkpoint outputs/models/dqn_latest.pt
    python scripts/debug_dqn_training.py --checkpoint outputs/models/dqn_latest.pt \
        --device auto --seed 42
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.rules import PLAYER1, PLAYER2, initial_board
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    encode_action,
    encode_state,
    inverse_transform_move_for_player,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel
from corners_rl.rl.replay_buffer import ReplayBuffer
from corners_rl.rl.self_play import compute_shaped_reward
from corners_rl.utils.seeding import resolve_device


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"


def _sep(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def _fail(msg: str) -> None:
    print(f"{FAIL} {msg}")


def _warn(msg: str) -> None:
    print(f"{WARN} {msg}")


def _info(msg: str) -> None:
    print(f"{INFO} {msg}")


def _weight_norm(model: DQNModel) -> float:
    total = sum(p.float().norm().item() ** 2 for p in model.parameters())
    return float(total ** 0.5)


# ── Section 1: Checkpoint ─────────────────────────────────────────────────────

def check_checkpoint(ckpt_path: Path, device: str) -> DQNAgent:
    """Load and inspect checkpoint; return a DQNAgent (random if no checkpoint)."""
    _sep("1. CHECKPOINT")

    if not ckpt_path.exists():
        _warn(f"Checkpoint not found: {ckpt_path}")
        _warn("Creating a RANDOM, untrained DQNAgent. Weights are random.")
        model = DQNModel()
        agent = DQNAgent(model=model, device=device, epsilon=0.0, seed=42)
        _info(f"Random model weight norm: {_weight_norm(model):.4f}")
        return agent

    _ok(f"Checkpoint found: {ckpt_path}  ({ckpt_path.stat().st_size // 1024} KB)")

    resolved = str(resolve_device(device))
    ckpt = torch.load(ckpt_path, map_location=resolved, weights_only=True)

    keys = list(ckpt.keys())
    _info(f"Keys in checkpoint: {keys}")

    if "model_state_dict" not in ckpt:
        _fail("'model_state_dict' key missing — cannot load weights!")
        model = DQNModel()
    else:
        _ok("'model_state_dict' key present")
        model = DQNModel()
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            _ok("model.load_state_dict() succeeded")
        except Exception as exc:
            _fail(f"model.load_state_dict() failed: {exc}")
            model = DQNModel()

    norm = _weight_norm(model)
    _info(f"Weight L2-norm: {norm:.4f}")
    if norm < 1.0:
        _warn("Very small weight norm — model may be near-zero init or undertrained.")

    saved_eps = ckpt.get("epsilon")
    if saved_eps is not None:
        _info(f"Saved epsilon: {saved_eps:.6f}")
        if saved_eps > 0.5:
            _warn(
                f"Saved epsilon={saved_eps:.3f} is high — training ended early "
                "or this is from a very short run. The DQN behaved mostly randomly."
            )
    else:
        _info("No 'epsilon' key in checkpoint.")

    episode   = ckpt.get("episode")
    tot_steps = ckpt.get("total_steps")
    _info(f"episode={episode}, total_steps={tot_steps}")
    if tot_steps is not None and tot_steps < 5_000:
        _warn(
            f"total_steps={tot_steps} is very small — model is undertrained. "
            "Expected ≥20 000 steps for any meaningful learning."
        )

    replay_type = ckpt.get("replay_type")
    _info(f"replay_type: {replay_type}")

    model.to(resolved)
    agent = DQNAgent(model=model, device=resolved, epsilon=0.0, seed=42)
    _ok(f"DQNAgent created with epsilon=0.0, device={resolved}")
    return agent


# ── Section 2: DQN action sanity ─────────────────────────────────────────────

def check_action_sanity(agent: DQNAgent, n: int = 20) -> None:
    """Verify all greedy moves are legal; check for repetitive move selection."""
    _sep("2. DQN ACTION SANITY")

    env = CornersEnv()
    env.reset()

    chosen_actions: list[int] = []
    all_legal = True

    for i in range(n):
        legal = env.legal_moves()
        move = agent.select_move(env)

        if move not in legal:
            _fail(f"Step {i}: selected move {move} is NOT in legal_moves!")
            all_legal = False
        else:
            player = env.current_player
            canonical = transform_move_for_player(move, player)
            aid = encode_action(canonical)
            chosen_actions.append(aid)

        # advance env (use a random move to keep the game going)
        env.step(env.legal_moves()[0])
        if env.is_terminal():
            env.reset()

    if all_legal:
        _ok(f"All {n} selected moves are legal.")
    else:
        _fail("Some selected moves are illegal!")

    unique = len(set(chosen_actions))
    _info(f"Unique action IDs chosen over {n} steps: {unique}/{n}")
    if unique == 1:
        _warn("DQN always picks the same action — Q-values may be degenerate (all equal).")

    # Show Q-value distribution for one state
    env2 = CornersEnv()
    env2.reset()
    player = env2.current_player
    legal = env2.legal_moves()
    canon = [transform_move_for_player(m, player) for m in legal]
    state_t = torch.from_numpy(encode_state(env2.board, player)).unsqueeze(0)
    mask = legal_action_mask(canon)

    agent._model.eval()
    with torch.no_grad():
        q = agent._model(state_t.to(agent._device)).squeeze(0).cpu()

    legal_qs = q[mask]
    _info(
        f"Q-values over {mask.sum()} legal actions: "
        f"min={legal_qs.min():.4f}  max={legal_qs.max():.4f}  "
        f"mean={legal_qs.mean():.4f}  std={legal_qs.std():.4f}"
    )
    if legal_qs.std() < 1e-4:
        _warn(
            "Q-value std ≈ 0 — all legal moves have nearly identical Q-values. "
            "Either model is untrained (near-zero weights) or degenerate."
        )
    else:
        _ok("Q-value std > 1e-4 — network differentiates between moves.")


# ── Section 3: Evaluation sanity ──────────────────────────────────────────────

def check_eval_sanity(agent: DQNAgent, seed: int, n: int = 20) -> None:
    """Play DQN vs Random (both sides); print per-game results."""
    _sep("3. EVALUATION SANITY")

    _info(f"DQN epsilon during eval: {agent.epsilon}")
    if agent.epsilon > 0.0:
        _warn(f"DQN epsilon={agent.epsilon:.3f} > 0.0 — not fully greedy!")

    for role, desc in [((agent, RandomAgent(seed=seed)), "DQN=P1 vs Random=P-1"),
                        ((RandomAgent(seed=seed), agent), "Random=P1 vs DQN=P-1")]:
        p1_agent, pm1_agent = role
        results = {"dqn_wins": 0, "dqn_draws": 0, "dqn_losses": 0}
        sample_games: list[str] = []

        for i in range(n):
            env = CornersEnv(max_moves=200)
            env.reset()
            move_map = {1: p1_agent, -1: pm1_agent}
            all_moves = []

            while not env.is_terminal():
                pl = env.current_player
                mv = move_map[pl].select_move(env)
                all_moves.append(mv)
                env.step(mv)

            winner = env.winner
            dqn_is_p1 = (p1_agent is agent)
            if winner is None:
                results["dqn_draws"] += 1
                outcome = "draw"
            elif (winner == 1 and dqn_is_p1) or (winner == -1 and not dqn_is_p1):
                results["dqn_wins"] += 1
                outcome = "DQN WIN"
            else:
                results["dqn_losses"] += 1
                outcome = "DQN LOSS"

            if i < 3:
                sample_games.append(
                    f"  Game {i+1}: winner={winner}, moves={len(all_moves)}, outcome={outcome}"
                )

        _info(f"\n  {desc}  ({n} games):")
        print(f"    DQN wins={results['dqn_wins']}  "
              f"draws={results['dqn_draws']}  "
              f"losses={results['dqn_losses']}")
        print("  Sample games (first 3):")
        for line in sample_games:
            print(line)

        if results["dqn_wins"] == 0 and results["dqn_losses"] == 0:
            _warn("DQN never won AND never lost — all draws (max_moves reached).")
        elif results["dqn_wins"] > 0:
            _ok(f"DQN won {results['dqn_wins']}/{n} games.")


# ── Section 4: Reward sanity ──────────────────────────────────────────────────

def check_reward_sanity() -> None:
    """Verify shaped reward signs and magnitudes."""
    _sep("4. REWARD SANITY")

    env = CornersEnv()
    env.reset()
    board = env.board
    player = PLAYER1
    legal = env.legal_moves()

    rewards_info: list[tuple[str, float]] = []
    for move in legal[:8]:
        from corners_rl.env.moves import apply_move
        board_after = apply_move(board, move, player)
        r = compute_shaped_reward(board, board_after, player, move, False, None)
        direction = "→ toward goal" if r > 0 else ("→ neutral/away" if r < -0.02 else "→ neutral")
        rewards_info.append((str(move), r))
        _info(f"  move={move}  reward={r:.4f}  {direction}")

    if rewards_info:
        rs = [r for _, r in rewards_info]
        if max(rs) > min(rs):
            _ok("Shaped reward differentiates between moves.")
        else:
            _warn("All moves produce the same shaped reward — reward may be constant.")

    # Check win reward
    from corners_rl.env.moves import apply_move
    board_win = np.zeros((8, 8), dtype=np.int8)
    # PLAYER1 wins by filling bottom-right 3x3
    from corners_rl.env.rules import get_target_zone
    for r, c in get_target_zone(PLAYER1):
        board_win[r, c] = PLAYER1
    # Pretend the last move was (5,5) → (7,7)
    dummy_move = ((5, 5), (7, 7))
    board_win_before = board_win.copy()
    r_win = compute_shaped_reward(board_win_before, board_win, PLAYER1, dummy_move, True, PLAYER1)
    _info(f"Terminal WIN reward for PLAYER1: {r_win:.2f}")
    if r_win > 50:
        _ok("Win reward is positive and large.")
    else:
        _fail(f"Win reward={r_win:.2f} is unexpectedly small!")

    r_lose = compute_shaped_reward(board_win_before, board_win, PLAYER2, dummy_move, True, PLAYER1)
    _info(f"Terminal LOSS reward for PLAYER2 (opponent): {r_lose:.2f}")
    if r_lose < -50:
        _ok("Loss reward is negative and large.")
    else:
        _fail(f"Loss reward={r_lose:.2f} is unexpectedly small/positive!")


# ── Section 5: Training step sanity ──────────────────────────────────────────

def check_training_step_sanity(device: str) -> None:
    """Verify that one gradient step changes weights and produces finite loss."""
    _sep("5. TRAINING STEP SANITY")

    from corners_rl.rl.train_dqn import dqn_update

    resolved = str(resolve_device(device))
    model = DQNModel().to(resolved)
    target = copy.deepcopy(model).to(resolved)
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fill a small replay buffer with random transitions
    buf = ReplayBuffer(capacity=200, seed=0)
    env = CornersEnv(max_moves=50)
    env.reset()

    while len(buf) < 100:
        if env.is_terminal():
            env.reset()
        player = env.current_player
        board_before = env.board
        legal = env.legal_moves()
        canon = [transform_move_for_player(m, player) for m in legal]
        state_arr = encode_state(board_before, player)

        move = legal[0]
        canonical_move = transform_move_for_player(move, player)
        action_id = encode_action(canonical_move)

        board_after, _, done, info = env.step(move)
        r = compute_shaped_reward(board_before, board_after, player, move, done, info["winner"])

        if done:
            from corners_rl.rl.encoding import BOARD_SIZE, STATE_CHANNELS
            next_arr = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            next_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        else:
            next_player = env.current_player
            next_arr = encode_state(board_after, next_player)
            next_canon = [transform_move_for_player(m, next_player) for m in env.legal_moves()]
            next_mask = legal_action_mask(next_canon)

        buf.push(state_arr, action_id, r, next_arr, done, next_mask)

    _ok(f"Replay buffer filled: {len(buf)} transitions")

    # Record weight norm before update
    norm_before = _weight_norm(model)

    losses: list[float] = []
    for step in range(10):
        batch = buf.sample(32)
        loss, td_errs = dqn_update(batch, model, target, optimizer, gamma=0.99,
                                   device=torch.device(resolved))
        losses.append(loss)

    norm_after = _weight_norm(model)
    weight_delta = abs(norm_after - norm_before)

    _info(f"Loss over 10 steps: min={min(losses):.4f}  max={max(losses):.4f}  "
          f"last={losses[-1]:.4f}")
    _info(f"Weight norm: {norm_before:.4f} → {norm_after:.4f}  (Δ={weight_delta:.6f})")

    if any(not np.isfinite(l) for l in losses):
        _fail("Loss is NaN or Inf — training is diverging!")
    else:
        _ok("All 10 loss values are finite.")

    if weight_delta < 1e-7:
        _fail("Weight norm did not change — optimizer.step() may not be working!")
    else:
        _ok(f"Weights changed after optimizer.step() (Δ={weight_delta:.6f}).")


# ── Section 6: Perspective/transform sanity ───────────────────────────────────

def check_perspective_sanity() -> None:
    """Verify encode_state, transform, inverse_transform, action_id_to_move."""
    _sep("6. PERSPECTIVE / TRANSFORM SANITY")

    from corners_rl.rl.encoding import action_id_to_move, decode_action

    board = initial_board()

    state_p1  = encode_state(board, PLAYER1)
    state_p2  = encode_state(board, PLAYER2)

    _info(f"state_p1 shape: {state_p1.shape}  dtype: {state_p1.dtype}")
    _info(f"state_p2 shape: {state_p2.shape}  dtype: {state_p2.dtype}")

    # P1 pieces in top-left corner → channel 0 should be 1 at (0,0)..(2,2)
    if state_p1[0, 0, 0] == 1.0:
        _ok("PLAYER1 state: own pieces at canonical top-left (correct).")
    else:
        _fail("PLAYER1 state: own pieces NOT at top-left!")

    # P2 canonical = rotation: P2 pieces originally at bottom-right → after
    # rotation they appear at top-left in the canonical frame
    if state_p2[0, 0, 0] == 1.0:
        _ok("PLAYER2 state: own pieces at canonical top-left after 180° rotation (correct).")
    else:
        _fail("PLAYER2 state: own pieces NOT at top-left in canonical frame!")

    # Transform round-trip
    env = CornersEnv()
    env.reset()
    legal_p1 = env.legal_moves()
    env.step(legal_p1[0])
    legal_p2 = env.legal_moves()

    for player, legal in [(PLAYER1, legal_p1), (PLAYER2, legal_p2)]:
        for move in legal[:5]:
            canonical = transform_move_for_player(move, player)
            restored  = inverse_transform_move_for_player(canonical, player)
            if restored == move:
                pass
            else:
                _fail(f"Round-trip failed for player={player}: {move} → {canonical} → {restored}")
                break
        else:
            _ok(f"Player {player}: transform → inverse_transform round-trip OK for 5 moves.")

    # action_id_to_move never returns an illegal action
    env2 = CornersEnv()
    env2.reset()
    for player, legal in [(PLAYER1, env2.legal_moves())]:
        canon = [transform_move_for_player(m, player) for m in legal]
        mask = legal_action_mask(canon)
        legal_ids = np.where(mask)[0]
        for aid in legal_ids[:10]:
            move = action_id_to_move(aid, canon)
            # Inverse-transform back and check it's in original legal_moves
            real_move = inverse_transform_move_for_player(move, player)
            if real_move not in legal:
                _fail(f"action_id_to_move({aid}) → {move} → {real_move} not in legal_moves!")
                break
        else:
            _ok("action_id_to_move → inverse_transform always lands in legal_moves.")


# ── Section 7: Summary calculation sanity ────────────────────────────────────

def check_summary_sanity(agent: DQNAgent, seed: int) -> None:
    """Verify winner_agent is determined correctly when sides swap."""
    _sep("7. SUMMARY CALCULATION SANITY")

    from corners_rl.evaluation.evaluate import evaluate_match, summarize_results

    random_agent = RandomAgent(name="random", seed=seed)
    dqn = agent  # name should be "dqn"

    _info(f"DQN agent name: '{dqn.name}'")
    _info(f"Random agent name: '{random_agent.name}'")

    df = evaluate_match(dqn, random_agent, games=4, max_moves=200, seed=seed)

    _info("  game_id | player1_agent | player_-1_agent | winner | winner_agent")
    for _, row in df.iterrows():
        print(
            f"    {int(row['game_id'])}      | {row['player1_agent']:<14} | "
            f"{row['player_minus1_agent']:<14} | {str(row['winner']):<6} | "
            f"{str(row['winner_agent'])}"
        )

    # Check: when DQN plays as P1 and wins (winner==1), winner_agent should be 'dqn'
    p1_dqn_rows = df[df["player1_agent"] == dqn.name]
    for _, row in p1_dqn_rows.iterrows():
        if row["winner"] == 1:
            if row["winner_agent"] == dqn.name:
                _ok(f"Game {row['game_id']}: DQN=P1 wins → winner_agent='{row['winner_agent']}' ✓")
            else:
                _fail(
                    f"Game {row['game_id']}: DQN=P1 wins → winner_agent='{row['winner_agent']}' "
                    f"(expected '{dqn.name}')"
                )
        elif row["winner"] == -1:
            if row["winner_agent"] == random_agent.name:
                _ok(
                    f"Game {row['game_id']}: DQN=P1 loses → winner_agent='{row['winner_agent']}' ✓"
                )
            else:
                _fail(
                    f"Game {row['game_id']}: DQN=P1 loses → winner_agent='{row['winner_agent']}' "
                    f"(expected '{random_agent.name}')"
                )

    summary = summarize_results(df)
    _info(
        f"evaluate_match summary: "
        f"agent1_win_rate={summary['agent1_win_rate']:.2f}  "
        f"agent2_win_rate={summary['agent2_win_rate']:.2f}  "
        f"draw_rate={summary['draw_rate']:.2f}"
    )
    total_check = summary["agent1_win_rate"] + summary["agent2_win_rate"] + summary["draw_rate"]
    if abs(total_check - 1.0) < 1e-6:
        _ok("win_rate + opponent_win_rate + draw_rate = 1.0 ✓")
    else:
        _fail(f"Rates don't sum to 1.0: {total_check:.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose DQN training pipeline issues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=Path("outputs/models/dqn_latest.pt"),
        help="Path to DQN checkpoint (.pt).",
    )
    p.add_argument("--device", default="auto", help="Torch device.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print("  DQN Training Pipeline — Diagnostic Report")
    print(f"{'='*60}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {args.device}")
    print(f"  seed       : {args.seed}")

    agent = check_checkpoint(args.checkpoint, args.device)
    check_action_sanity(agent)
    check_eval_sanity(agent, seed=args.seed)
    check_reward_sanity()
    check_training_step_sanity(args.device)
    check_perspective_sanity()
    check_summary_sanity(agent, seed=args.seed)

    print(f"\n{'='*60}")
    print("  Diagnostic complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
