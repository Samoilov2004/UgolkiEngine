"""Reinforcement-learning utilities: encoding, model, replay buffer."""

from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    STATE_CHANNELS,
    action_id_to_move,
    decode_action,
    encode_action,
    encode_state,
    inverse_transform_move_for_player,
    inverse_transform_pos_for_player,
    legal_action_mask,
    transform_move_for_player,
    transform_pos_for_player,
)
from corners_rl.rl.model import DQNModel, masked_argmax
from corners_rl.rl.replay_buffer import ReplayBuffer

__all__ = [
    # encoding
    "BOARD_SIZE",
    "STATE_CHANNELS",
    "ACTION_SPACE_SIZE",
    "encode_state",
    "encode_action",
    "decode_action",
    "legal_action_mask",
    "action_id_to_move",
    "transform_pos_for_player",
    "inverse_transform_pos_for_player",
    "transform_move_for_player",
    "inverse_transform_move_for_player",
    # model
    "DQNModel",
    "masked_argmax",
    # buffer
    "ReplayBuffer",
]
