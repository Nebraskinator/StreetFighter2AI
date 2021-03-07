import random
from itertools import zip_longest
from typing import List

from config import MuZeroConfig
from game.game import AbstractGame
import _pickle as cPickle
import os
import numpy as  np

class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.loaded_games = []
        self.current_games = []
        self.memory_path = config.memory_path

    def save_game(self, game):
        if sum([len(i.root_values) for i in self.buffer])  > self.window_size:
            self.buffer.pop(0)
            
        game.game_priority = 1e3*len(game.root_values)
        game.priorities = list(np.full(len(game.root_values), 1e3))
        self.buffer.append(game)
        

            
    def update_buffer(self):
        new_files = [f for f in os.listdir(self.memory_path) if f not in self.loaded_games]
        new_files.sort(key = lambda x: int(x.split('_')[1]))
        if len(new_files) > self.window_size // 250:
            self.loaded_games = self.loaded_games + new_files[:-self.window_size // 250]
            new_files = new_files[-self.window_size // 250:]
        if len(new_files) != 0:
            for new_file in new_files:
                with open(os.path.join(self.memory_path,new_file), 'rb') as game_file:
                    game = cPickle.load(game_file)
                self.save_game(game)
                self.loaded_games.append(new_file)
                if sum([len(i.root_values) for i in self.buffer]) > self.window_size: 
                    self.current_games.pop(0)
                self.current_games.append(new_file)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        # Generate some sample of data to train on
        games = self.sample_games()
        game_pos = [(g, self.sample_position(self.buffer[g])) for g in games]
        game_data = [(self.buffer[g].make_image(i), [action.index for action in self.buffer[g].history[i:i + num_unroll_steps]],
                      self.buffer[g].make_target(i, num_unroll_steps, td_steps, self.buffer[g].to_play()))
                     for (g, i) in game_pos]
        sample_weights = [self.buffer[g].priorities[i] for (g, i) in game_pos]
        game_weights = [self.buffer[g].game_priority for (g, i) in game_pos]
        weight_batch = 1 / (np.array(sample_weights) * np.array(game_weights))
        weight_batch = weight_batch / np.max(weight_batch)
        # Pre-process the batch
        image_batch, actions_time_batch, targets_batch = zip(*game_data)
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=0))

        # Building batch of valid actions and a dynamic mask for hidden representations during BPTT

        batch = image_batch, targets_init_batch, targets_time_batch, actions_time_batch
        return batch, game_pos, weight_batch**0.4

    def sample_games(self) -> List[AbstractGame]:
        # Sample game from buffer either uniformly or according to some priority.
        game_probs = np.array([game.game_priority for game in self.buffer])
        game_probs /= np.sum(game_probs)
        return np.random.choice(len(self.buffer), size=self.batch_size, p = game_probs)

    def sample_position(self, game: AbstractGame) -> int:
        # Sample position from game either uniformly or according to some priority.
        pos_probs = game.priorities / sum(game.priorities)
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
    def update_priorities(self, priorities, idx_info):
        for i in range(len(idx_info)):
            game_id, game_pos = idx_info[i]
            priority = priorities[i,:]
            start_idx = game_pos
            end_idx = min(game_pos+len(priority), len(self.buffer[game_id].priorities))
            self.buffer[game_id].priorities[start_idx:end_idx] = priority[:end_idx-start_idx]
            self.buffer[game_id].game_priority = np.mean(self.buffer[game_id].priorities) * len(self.buffer[game_id].root_values)