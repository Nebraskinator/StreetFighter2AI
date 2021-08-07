import random
from itertools import zip_longest
from typing import List

from config import MuZeroConfig
from game.game import AbstractGame
import _pickle as cPickle
import os
import numpy as  np

class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig, fighter):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.loaded_games = []
        self.current_games = []
        self.memory_path = config.memory_path
        self.fighter = fighter

    def save_game(self, game):
        if sum([len(i.root_values) for i in self.buffer])  > self.window_size:
            self.buffer.pop(0)
        if game.player1_historic_network == True: 
            game.game_player1_priority = 0
            game.player1_priorities = list(np.full(len(game.root_values), 0))
        else:
            game.game_player1_priority = 1e3*len(game.root_values)
            game.player1_priorities = list(np.full(len(game.root_values), 1e3))
            player1_zero_move_idx = [i for i, j in enumerate(game.child_visits) if j[0][0] == 1.]
            for idx in player1_zero_move_idx:
                game.player1_priorities[idx] = 0  
        if game.player2_historic_network == True: 
            game.game_player2_priority = 0
            game.player2_priorities = list(np.full(len(game.root_values), 0))
        else:                
            game.game_player2_priority = 1e3*len(game.root_values)
            game.player2_priorities = list(np.full(len(game.root_values), 1e3))
            player2_zero_move_idx = [i for i, j in enumerate(game.child_visits) if j[1][0] == 1.]
            for idx in player2_zero_move_idx:
                game.player2_priorities[idx] = 0        
        self.buffer.append(game)

            
    def update_buffer(self):
        new_files = [f for f in os.listdir(self.memory_path) if f not in self.loaded_games]
        new_files = [f for f in new_files if (f.split('_')[-1][:-4] == self.fighter) | (f.split('_')[-2] == self.fighter)]
        new_files.sort(key = lambda x: int(x.split('_')[1]))
        if len(new_files) > self.window_size // 1100:
            self.loaded_games = self.loaded_games + new_files[:-self.window_size // 1100]
            new_files = new_files[-self.window_size // 1100:]
        if len(new_files) != 0:
            for new_file in new_files:
                with open(os.path.join(self.memory_path,new_file), 'rb') as game_file:
                    game = cPickle.load(game_file)
                self.save_game(game)
                self.loaded_games.append(new_file)
                if sum([len(i.root_values) for i in self.buffer]) > self.window_size: 
                    self.current_games.pop(0)
                self.current_games.append(new_file)

    def sample_batch(self, num_unroll_steps: int, unroll_step_size : int, td_steps: int, fighter):
        # Generate some sample of data to train on
        games = self.sample_games(fighter)
        game_pos = [(g, self.sample_position(self.buffer[g], fighter), 'player1' if self.buffer[g].player1 == fighter else 'player2') for g in games]
        game_data = [(self.buffer[g].make_image(i, p), [action.index for action in [j[int(p[-1]) - 1] for j in self.buffer[g].history[i:i + num_unroll_steps]]],
                      self.buffer[g].make_target(i, num_unroll_steps, unroll_step_size, td_steps, p))
                     for (g, i, p) in game_pos]
        sample_weights = [self.buffer[g].player1_priorities[i] if p == 'player1' else self.buffer[g].player2_priorities[i] for (g, i, p) in game_pos]
        game_weights = [self.buffer[g].game_player1_priority if p == 'player1' else self.buffer[g].game_player2_priority for (g, i, p) in game_pos]
        weight_batch = 1 / (np.array(sample_weights) * np.array(game_weights))
        weight_batch = weight_batch / np.max(weight_batch)
        # Pre-process the batch
        image_batch, actions_time_batch, targets_batch = zip(*game_data)
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=0))

        # Building batch of valid actions and a dynamic mask for hidden representations during BPTT

        batch = image_batch, targets_init_batch, targets_time_batch, actions_time_batch
        return batch, game_pos, weight_batch**0.4

    def sample_games(self, fighter) -> List[AbstractGame]:
        # Sample game from buffer either uniformly or according to some priority.
        game_probs = np.array([game.game_player1_priority if game.player1 == fighter else game.game_player2_priority for game in self.buffer])
        game_probs /= np.sum(game_probs)
        return np.random.choice(len(self.buffer), size=self.batch_size, p = game_probs)

    def sample_position(self, game: AbstractGame, fighter) -> int:
        # Sample position from game either uniformly or according to some priority.
        if game.player1 == fighter:
            pos_probs = game.player1_priorities / sum(game.player1_priorities)
        if game.player2 == fighter:
            pos_probs = game.player2_priorities / sum(game.player2_priorities)
        return np.random.choice(len(pos_probs), p=pos_probs)

    def sample_position_value_bias(self, game: AbstractGame) -> int:
        # Sample position from game either uniformly or according to some priority.
        history = [i.index for i in game.history]
        counts = np.bincount(history)
        common = np.argmax(counts)
        above_avg = [i[0] for i in np.argwhere(history==common)]
        below_avg = [i[0] for i in np.argwhere(history!=common)]
        if random.randint(0,5) != 5:
            return np.random.choice(below_avg)
        else:
            return np.random.choice(above_avg)
    def update_priorities(self, priorities, idx_info, fighter):
        for i in range(len(idx_info)):
            game_id, game_pos, _ = idx_info[i]
            priority = priorities[i,:]
            start_idx = game_pos
            
            if self.buffer[game_id].player1 == fighter:
                end_idx = min(game_pos+len(priority), len(self.buffer[game_id].player1_priorities))
                self.buffer[game_id].player1_priorities[start_idx:end_idx] = priority[:end_idx-start_idx]
                self.buffer[game_id].game_player1_priority = np.mean(self.buffer[game_id].player1_priorities) * len(self.buffer[game_id].root_values)
            if self.buffer[game_id].player2 == fighter:
                end_idx = min(game_pos+len(priority), len(self.buffer[game_id].player2_priorities))
                self.buffer[game_id].player2_priorities[start_idx:end_idx] = priority[:end_idx-start_idx]
                self.buffer[game_id].game_player2_priority = np.mean(self.buffer[game_id].player2_priorities) * len(self.buffer[game_id].root_values)
                
                                
                
                
                