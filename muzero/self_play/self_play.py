"""Self-Play module: where the games are played."""

from config import MuZeroConfig
from game.game import AbstractGame, Action
from networks.network import AbstractNetwork
from networks.shared_storage import SharedStorage
from self_play.mcts import run_mcts, select_action, expand_node, add_exploration_noise
from self_play.utils import Node
from training.replay_buffer import ReplayBuffer
import random
import numpy as np


def run_selfplay(config: MuZeroConfig, player1_storage: SharedStorage, player2_storage: SharedStorage, game_name: str):
    """Take the latest network, produces multiple games and save them in the shared replay buffer"""



    game = play_game(config, player1_storage, player2_storage, game_name=game_name)
    game.env = 0
    rewards = int(sum([i[0] for i in game.rewards]) - sum(i[1] for i in game.rewards) / 2)
    game.save_gif(game_name+'_'+str(rewards)+'_'+game.player1+'_'+game.player2)
    game.gif_images = 0
    game.save_game_to_file(game_name+'_'+str(rewards)+'_'+game.player1+'_'+game.player2+'.pkl')
    
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


def play_game(config: MuZeroConfig, player1_storage: SharedStorage, player2_storage: SharedStorage, train: bool = True, game_name: str = 'default_agent') -> AbstractGame:
    """
    Each game is produced by starting at the initial board position, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.
    """
    game = config.new_game()
    

    dice = np.random.rand()
    if dice < 0.1:
        game.player1_historic_network = True
        game.player1_noop = True
        game.player2_noop = False
    elif dice < 0.2:
        game.player2_historic_network = True
        game.player1_noop = False            
        game.player2_noop = True             
    else:
        game.player1_noop = False
        game.player2_noop = False
    if int(game_name.split('_')[-1]) < 25:
        player1_network = player1_storage.uniform_network
        player2_network = player2_storage.uniform_network
        print('Initial training: using uniform network')
    else:
        if game.player1_noop == True:
            player1_network = player1_storage.uniform_network
            player2_network = player2_storage.latest_network(game.player2)        
        elif game.player2_noop == True:
            player2_network = player2_storage.uniform_network
            player1_network = player1_storage.latest_network(game.player1)   
        else:
            player2_network = player2_storage.latest_network(game.player2) 
            player1_network = player1_storage.latest_network(game.player1)
    mode_action_select = 'softmax' if train else 'max'
    noop_action = np.random.randint(7)
    noop = random.randint(0,config.initial_random_moves)
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        target_policies = [] 

        if game.player1_ready == True:
            current_observation_player1 = game.make_image(-1,'player1')
            inital_network_output_player1 = player1_network.initial_inference(current_observation_player1)
            player1_action_values = value_search(config, player1_network, inital_network_output_player1)

            min_value = sorted(player1_action_values, key=player1_action_values.get)[0]
            min_value = player1_action_values[min_value]
            max_value = sorted(player1_action_values, key=player1_action_values.get)[-1]
            max_value = player1_action_values[max_value]
            target_policy = []
            normalized_action_values = {}
            for action in inital_network_output_player1.policy_logits.keys():
                current_value = (player1_action_values[action] - min_value + 0.00001) / (max_value - min_value + 0.00001)
                target_policy.append(current_value)
                normalized_action_values[action] = current_value
            target_sum = sum(target_policy)
            target_policy = [i/target_sum for i in target_policy]
            target_policies.append(target_policy)            
            player1_action_values = np.mean(list(player1_action_values.values()))
            if game.player1_noop == True:
                player1_action = Action(noop_action)
            else:
                player1_action = select_action(config, len(game.history), normalized_action_values, mode=mode_action_select)

        else:
            try:
                player1_action_values = game.root_values[-1][0]
            except:
                player1_action_values = 0
            target_policy = np.zeros(len(game.actions))
            target_policy[0] = 1
            target_policies.append(target_policy)
            player1_action = Action(1)
        if game.player2_ready == True:
            current_observation_player2 = game.make_image(-1,'player2')
            inital_network_output_player2 = player2_network.initial_inference(current_observation_player2)
            player2_action_values = value_search(config, player2_network, inital_network_output_player2)
            
            min_value = sorted(player2_action_values, key=player2_action_values.get)[0]
            min_value = player2_action_values[min_value]
            max_value = sorted(player2_action_values, key=player2_action_values.get)[-1]
            max_value = player2_action_values[max_value]
            target_policy = []
            normalized_action_values = {}
            for action in inital_network_output_player2.policy_logits.keys():
                current_value = (player2_action_values[action] - min_value + 0.00001) / (max_value - min_value + 0.00001)
                target_policy.append(current_value)
                normalized_action_values[action] = current_value
            target_sum = sum(target_policy)
            target_policy = [i/target_sum for i in target_policy]
            target_policies.append(target_policy)    
            player2_action_values = np.mean(list(player2_action_values.values()))
            if game.player2_noop == True:
                player2_action = Action(noop_action)
            else:            
                player2_action = select_action(config, len(game.history), normalized_action_values, mode=mode_action_select)

        else:
            try:
                player2_action_values = game.root_values[-1][1]
            except:
                player2_action_values = 0
            target_policy = np.zeros(len(game.actions))
            target_policy[0] = 1
            target_policies.append(target_policy)
            player2_action = Action(1)
        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        game.root_values.append([player1_action_values,player2_action_values])
        game.child_visits.append(target_policies)

        game.apply(player1_action,player2_action)

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
            
            
            
            
            

