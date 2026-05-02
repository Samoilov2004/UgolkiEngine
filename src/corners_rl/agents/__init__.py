"""Agent implementations."""

from corners_rl.agents.base import BaseAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.agents.greedy_agent import GreedyAgent, total_distance
from corners_rl.agents.heuristic_agent import HeuristicAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "HeuristicAgent",
    "total_distance",
    # DQNAgent is imported lazily to avoid a hard torch dependency at import time
]
