import collections
from typing import Optional, Dict

import tensorflow as tf

from game.cartpole import StreetFighter2
from game.game import AbstractGame
from networks.cartpole_network import StreetFighter2Network
from networks.network import BaseNetwork, UniformNetwork

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):

    def __init__(self,
                 game,
                 nb_training_loop: int,
                 nb_episodes: int,
                 nb_epochs: int,
                 network_args: Dict,
                 network,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 visit_softmax_temperature_fn,
                 lr: float,
                 network_path: str,
                 memory_path: str,
                 known_bounds: Optional[KnownBounds] = None,
                 ):
        ### Environment
        self.game = game

        ### Self-Play
        self.action_space_size = action_space_size
        # self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.35

        # UCB formula
        self.pb_c_base = 19652//5
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.nb_training_loop = nb_training_loop
        self.nb_episodes = nb_episodes  # Nb of episodes per training loop
        self.nb_epochs = nb_epochs  # Nb of epochs per training loop

        # self.training_steps = int(1000e3)
        # self.checkpoint_interval = int(1e3)
        self.window_size = int(110000)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.unroll_step_size = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.network_args = network_args
        self.network = network
        self.lr = lr
        
        self.initial_random_moves = 0
        self.num_random_series = 0
        self.random_series_len = 10
        self.random_series_window = 500
        
        self.network_path = network_path
        self.memory_path = memory_path
        
        self.num_searches = 12
        self.search_depth = 2
        # Exponential learning rate schedule
        # self.lr_init = lr_init
        # self.lr_decay_rate = 0.1
        # self.lr_decay_steps = lr_decay_steps

    def new_game(self) -> AbstractGame:
        return self.game(self.discount, self.memory_path)

    def new_network(self) -> BaseNetwork:
        return self.network(**self.network_args)

    def uniform_network(self) -> UniformNetwork:
        return UniformNetwork(self.action_space_size)

    def new_optimizer(self) -> tf.keras.optimizers:
        return tf.keras.optimizers.Adam(learning_rate=self.lr, clipvalue=5)
    def ex_optimizer(self) -> tf.keras.optimizers:
        return tf.keras.optimizers.Adam(learning_rate=self.lr/4, clipvalue=5)
#        return tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum)  
    


def make_streetfighter2_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 120:
            return 4
        elif training_steps < 240:
            return 5
        elif training_steps < 360:
            return 6
        elif training_steps < 480:
            return 7
        elif training_steps < 600:
            return 8
        elif training_steps < 720:
            return 9
        else:
            return 9


    return MuZeroConfig(
        game=StreetFighter2,
        nb_training_loop=5000,
        nb_episodes=10,
        nb_epochs=500,
        network_args={'action_size': 36,
                      'state_size':  (96,96,64),
                      'representation_size': 256,
                      'min_value' : -(130//4+1),
                      'max_value': 130//4+1},
        network=StreetFighter2Network,
        action_space_size=36,
        max_moves=3300,
        discount=0.97,
        dirichlet_alpha=0.25,
        num_simulations=11,  # Odd number perform better in eval mode
        batch_size=64,
        td_steps=60,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        lr=0.0004,
        network_path = r'D:/StreetFighter2Tournament/Networks',
        memory_path = r'D:/StreetFighter2Tournament/Memories',
        )


"""
Legacy configs from the DeepMind's pseudocode.

def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=800,
        batch_size=2048,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1))


def make_go_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


def make_chess_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


def make_shogi_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)


def make_atari_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature)
"""
