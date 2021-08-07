from config import MuZeroConfig, make_streetfighter2_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network
import threading
import time
import os
import numpy as np

"""
MuZero training is split into two independent parts: Network training and
self-play data generation.
These two parts only communicate by transferring the latest networks checkpoint
from the training to the self-play, and the finished games from the self-play
to the training.
In contrast to the original MuZero algorithm this version doesn't works with
multiple threads, therefore the training and self-play is done alternately.
"""
'''
config = make_spaceinvaders_config()
storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
'''
config = make_streetfighter2_config()
#replay_buffer = ReplayBuffer(config)
storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer(), config.ex_optimizer(), config.network_path)

                     
prev_loops = [int(f.split('_')[1]) for f in os.listdir(config.memory_path)]    
try:
    max_prev_loops = np.amax(prev_loops)
except:
    max_prev_loops = 0
    
for loop in range(3,5000000):
    print("Training loop", loop)
    config.loops = loop
    

    train_network(config, storage, config.nb_epochs, ['zangief','ehonda'], loop)


#        print("Train score:", score_train)
#        print("Eval score:", run_eval(config, storage, 1))
    print(f"MuZero played {4 * (loop)} "
          f"episodes and trained for {config.nb_epochs * (loop)} epochs.\n")


