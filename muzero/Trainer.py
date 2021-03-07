from config import make_supermariobros_config
from networks.shared_storage import SharedStorage
from training.replay_buffer import ReplayBuffer
from training.training import train_network
import os
import numpy as np


config = make_supermariobros_config()
replay_buffer = ReplayBuffer(config)
storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer(), config.ex_optimizer(), config.network_path)
storage.latest_network()

                     
prev_loops = [int(f.split('_')[1]) for f in os.listdir(config.memory_path)]    
try:
    max_prev_loops = np.amax(prev_loops)
except:
    max_prev_loops = 0
    
for loop in range(max_prev_loops+1,5000000):
    print("Training loop", loop)
    config.loops = loop
    

    train_network(config, storage, replay_buffer, config.nb_epochs)
    print(f"MuZero played {4 * (loop)} "
          f"episodes and trained for {config.nb_epochs * (loop)} epochs.\n")


