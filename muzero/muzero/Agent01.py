import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from config import make_supermariobros_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay
import numpy as np


config = make_supermariobros_config()
storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer(), config.ex_optimizer(), config.network_path)

agent = 'agent01'
      
prev_loops = [int(f.split('_')[1]) for f in os.listdir(config.memory_path)]    
try:
    loop = np.amax(prev_loops)
except:
    loop = 0
    
while loop < 5000:
    prev_loops = [int(f.split('_')[1]) for f in os.listdir(config.memory_path)]    
    try:
        max_prev_loops = np.amax(prev_loops)
    except:
        max_prev_loops = 0
    if max_prev_loops > loop:
        loop = max_prev_loops
    print("Training loop", loop)
    config.loops = loop
    score_train = run_selfplay(config, storage,agent+'_'+str(loop))
    print(str(score_train))
    loop+=1
