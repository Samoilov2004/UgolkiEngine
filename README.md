# Corners RL — Reinforcement Learning Agent for the Game of Corners (Уголки)

A research project training a DQN agent with self-play to play the classic board game **Уголки** (Corners).

## Game Rules

- 8×8 board, two players
- Each player has **9 pieces** placed in a 3×3 corner zone at game start
- **Player 1** starts in the top-left corner, goal is the bottom-right 3×3 zone
- **Player -1** starts in the bottom-right corner, goal is the top-left 3×3 zone
- **Legal moves — orthogonal only, no diagonal moves or jumps:**
  - Simple step: move to an orthogonally adjacent free cell (up / down / left / right, distance 1)
  - Jump: hop over an **occupied** orthogonally adjacent cell to the free cell immediately beyond it (distance 2)
  - Chain jump: a sequence of jumps in one turn; every segment must be an orthogonal jump
  - Jumped pieces are **not** captured; simple steps cannot be mixed into a chain
  - **Diagonal moves and jumps are strictly forbidden**
- Every move is validated by `validate_move()` before application — no agent or DQN inference can produce an illegal board state
- **Win condition:** all 9 of your pieces occupy the opponent's starting zone

## Project Structure

```
corners_rl/
├── configs/          # YAML configuration files
├── scripts/          # CLI entry points
└── src/
    └── corners_rl/
        ├── env/            # Game environment
        ├── agents/         # Agents (random, greedy, DQN)
        ├── models/         # Neural network architectures
        ├── training/       # Training loop, replay buffer, self-play
        ├── evaluation/     # Evaluation & tournament
        └── visualization/  # Board renderer & training plots
```

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start

### Verify game rules (orthogonality check)

```bash
python scripts/check_rules.py --games 50 --max-moves 300
```

### Train DQN agent with self-play

```bash
python scripts/train.py --config configs/dqn.yaml
```

### Evaluate agents (tournament)

```bash
python scripts/evaluate.py --checkpoint checkpoints/dqn_best.pt --episodes 200
```

### Watch a game

```bash
python scripts/play.py --agent1 dqn --agent2 random --checkpoint checkpoints/dqn_best.pt
```

### Render a GIF of a game

```bash
python scripts/play.py --agent1 dqn --agent2 greedy --render gif --output game.gif
```

## Configuration

All hyperparameters live in `configs/dqn.yaml`. Key sections:

| Section | Description |
|---|---|
| `env` | Board size, reward shaping |
| `agent` | DQN hyperparameters (lr, gamma, epsilon schedule) |
| `training` | Episodes, batch size, target-net update frequency |
| `self_play` | Pool size, swap frequency |
| `evaluation` | Evaluation interval, number of episodes |

## Development

```bash
# Run tests
pytest tests/

# Type-check
mypy src/

# Lint
ruff check src/
```

## References

- Mnih et al., 2015 — [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- Silver et al., 2017 — [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
