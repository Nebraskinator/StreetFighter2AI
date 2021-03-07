"""MCTS module: where MuZero thinks inside the tree."""

import math
import random
from typing import List

import numpy

from config import MuZeroConfig
from game.game import Player, Action, ActionHistory
from networks.network import NetworkOutput, BaseNetwork
from self_play.utils import MinMaxStats, softmax_sample

def select_action(config: MuZeroConfig, num_moves: int, action_values: dict, mode: str = 'softmax'):
    """
    After running simulations inside in MCTS, we select an action based on the root's children visit counts.
    During training we use a softmax sample for exploration.
    During evaluation we select the most visited child.
    """
    visit_counts = [value for value in action_values.values()]
    actions = [action for action in action_values.keys()]
    action = None
    if mode == 'softmax':
        t = config.visit_softmax_temperature_fn(
            num_moves=num_moves, training_steps=config.loops)
        action = softmax_sample(visit_counts, actions, t)
    elif mode == 'max':
        action = actions[numpy.argmax(visit_counts)]
    elif mode == 'random':
        i = random.randint(0,config.action_space_size - 1)
        action = actions[i]
    return action
