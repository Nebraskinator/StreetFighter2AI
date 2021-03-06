import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from config import MuZeroConfig, make_streetfighter2_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network
import threading
import time

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
player1_storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer(), config.ex_optimizer(), config.network_path)
player2_storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer(), config.ex_optimizer(), config.network_path)

agent = 'agent01'
      
prev_loops = [int(f.split('_')[1]) for f in os.listdir(config.memory_path)]    
try:
    loop = np.amax(prev_loops)
except:
    loop = 0
    
while loop < 50000:
    prev_loops = [int(f.split('_')[1]) for f in os.listdir(config.memory_path)]    
    try:
        max_prev_loops = np.amax(prev_loops)
    except:
        max_prev_loops = 0
    if max_prev_loops > loop:
        loop = max_prev_loops
    print("Training loop", loop)
    config.loops = loop
    score_train = run_selfplay(config, player1_storage, player2_storage,agent+'_'+str(loop))
    print(str(score_train))
    loop+=1
