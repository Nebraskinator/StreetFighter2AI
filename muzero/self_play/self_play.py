"""Self-Play module: where the games are played."""

from config import MuZeroConfig
from game.game import AbstractGame
from networks.network import AbstractNetwork
from networks.shared_storage import SharedStorage
from self_play.mcts import select_action
import random
import numpy as np


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, game_name: str):
    """Take the latest network, produces multiple games and save them in the shared replay buffer"""


    if int(game_name.split('_')[-1]) < 100: #We want to run random games initially to fill the replay buffer
        network = storage.uniform_network
        print('Initial training: using uniform network')
    else:
        network = storage.latest_network()
    game = play_game(config, network, game_name=game_name)
    game.env = 0
    rewards = int(sum(game.rewards))
    game.save_game_to_file(game_name+'_'+str(rewards)+'.pkl')
    game.save_gif(game_name+'_'+str(rewards))
    return rewards


def run_eval(config: MuZeroConfig, storage: SharedStorage, eval_episodes: int):
    """Evaluate MuZero without noise added to the prior of the root and without softmax action selection"""
    network = storage.latest_network()
    returns = []
    for _ in range(eval_episodes):
        game = play_game(config, network, train=False)
        game.save_gif('eval')
        returns.append(sum(game.rewards))
    return sum(returns)*5 / eval_episodes if eval_episodes else 0


def play_game(config: MuZeroConfig, network: AbstractNetwork, train: bool = True, game_name: str = 'default_agent') -> AbstractGame:
    """
    Each game is produced by starting at the initial board position, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.
    """
    overworlds = ['1-1','2-1','3-1','3-2','4-1','5-1','5-2','6-1','6-2','7-1','8-1','8-2','8-3']
    underworlds = ['1-2','4-2']
    athletics = ['1-3','2-3','3-3','4-3','5-3','6-3','7-3']
    waterworlds = ['2-2','7-2']
    castles = ['1-4', '2-4', '3-4', '5-4', '6-4']
    game = config.new_game(overworlds)
    mode_action_select = 'softmax' if train else 'max'
    noop = random.randint(0,config.initial_random_moves)
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        current_observation = game.make_image(-1)
        inital_network_output = network.initial_inference(current_observation)
        action_values = value_search(config, network, inital_network_output)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        game.root_values.append(np.mean(list(action_values.values())))
        
        min_value = sorted(action_values, key=action_values.get)[0]
        min_value = action_values[min_value]
        max_value = sorted(action_values, key=action_values.get)[-1]
        max_value = action_values[max_value]
        target_policy = []
        normalized_action_values = {}
        for action in inital_network_output.policy_logits.keys():
            current_value = (action_values[action] - min_value + 0.001) / (max_value - min_value + 0.001)
            target_policy.append(current_value)
            normalized_action_values[action] = current_value
            
        target_sum = sum(target_policy)
        target_policy = [i/target_sum for i in target_policy]
        
        game.child_visits.append(target_policy)
        
        if len(game.history) < noop:
            action = select_action(config, len(game.history), normalized_action_values, mode='random')
        else:
            action = select_action(config, len(game.history), normalized_action_values, mode=mode_action_select)
        game.apply(action)

    game.close()
    return game

def value_search(config: MuZeroConfig, network: AbstractNetwork, inital_network_output):
    search_policy = inital_network_output.policy_softmax
    ranked_actions = sorted(search_policy, key=search_policy.get, reverse=True)
    actions_to_evaluate = ranked_actions[:config.num_searches]
    actions_to_infer = ranked_actions[config.num_searches:]
    action_values = {}
    for action in actions_to_evaluate:
        current_rewards = []
        current_values = [inital_network_output.value]
        current_hidden_state = inital_network_output.hidden_state
        current_action = action
        for i in range(config.search_depth):
            network_output = network.recurrent_inference(current_hidden_state, current_action)
            current_rewards.append(network_output.reward*config.discount**i)
            current_values.append(network_output.value*config.discount**i)
            current_hidden_state = network_output.hidden_state
            current_action = sorted(network_output.policy_softmax, key=network_output.policy_softmax.get, reverse=True)[0]
        action_values[action] = np.mean(current_values) + sum(current_rewards)
    min_value = min(action_values.values())
    for action in actions_to_infer:
        action_values[action] = min_value
    return action_values
            
            
            
            
            

