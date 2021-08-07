import tensorflow as tf
import os
from networks.network import BaseNetwork, UniformNetwork, AbstractNetwork
import pandas as pd
import numpy as np

class SharedStorage(object):
    """Save the different versions of the network."""

    def __init__(self, network: BaseNetwork, uniform_network: UniformNetwork, optimizer: tf.keras.optimizers, ex_optimizer: tf.keras.optimizers, network_path: str):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer
        self.ex_optimizer = ex_optimizer
        self.network_path = network_path
        self.training_loop = 0

    def latest_network(self, fighter) -> AbstractNetwork:
        '''
        if self._networks:
            step = self._networks[max(self._networks.keys())]
            self.current_network.representation_network.set_weights(self._representation_weights[step])
            self.current_network.value_network.set_weights(self._value_weights[step])
            self.current_network.policy_network.set_weights(self._policy_weights[step])
            self.current_network.dynamic_network.set_weights(self._dynamic_weights[step])
            self.current_network.reward_network.set_weights(self._reward_weights[step])
            return self.current_network
        '''        
#       else:
            # policy -> uniform, value -> 0, reward -> 0
            #return self.uniform_network
            
        networks_df = pd.read_csv(os.path.join(self.network_path,'current_networks.csv'))
        current_network = str(networks_df[fighter+'_current_network'].iloc[0])
        training_loop = int(networks_df['training_loop'].iloc[0])
        if training_loop > self.training_loop:
            self.training_loop = training_loop
        try:
            self.current_network.representation_network.load_weights(os.path.join(self.network_path, fighter+current_network+'_representation_network.h5'))
            self.current_network.value_network.load_weights(os.path.join(self.network_path, fighter+current_network+'_value_network.h5'))
            self.current_network.policy_network.load_weights(os.path.join(self.network_path, fighter+current_network+'_policy_network.h5'))
            self.current_network.dynamic_network.load_weights(os.path.join(self.network_path, fighter+current_network+'_dynamic_network.h5'))
            self.current_network.reward_network.load_weights(os.path.join(self.network_path, fighter+current_network+'_reward_network.h5'))
            self.current_network.exploration_network.load_weights(os.path.join(self.network_path, fighter+current_network+'_exploration_network.h5'))
            fighter_optimizer = np.load(os.path.join(self.network_path, fighter+current_network+'_optimizer.npy'), allow_pickle=True)
            self.current_network.optimizer_weights =  fighter_optimizer  
            print('Found previous network weights, pre-loading MuZero with saved weights.')
        except:
            print('No previous weights found, initializing model')
                            

        return self.current_network        

    def historic_network(self, fighter):
        networks_df = pd.read_csv(os.path.join(self.network_path,'current_networks.csv'))
        current_network = str(networks_df[fighter+'_current_network'].iloc[0])   
        historic_network_list = [f for f in os.listdir(self.network_path) if (f.startswith(fighter)) & (f.split('_')[1] == 'representation')]
        historic_network_list = [f.split('_')[0].strip(fighter) for f in historic_network_list]
        historic_network_list = [f for f in historic_network_list if f != current_network]
        if len(historic_network_list) == 0:
            historic_network = current_network
        else:
            historic_network = np.random.choice(historic_network_list)
        try:
            self.current_network.representation_network.load_weights(os.path.join(self.network_path, fighter+historic_network+'_representation_network.h5'))
            self.current_network.value_network.load_weights(os.path.join(self.network_path, fighter+historic_network+'_value_network.h5'))
            self.current_network.policy_network.load_weights(os.path.join(self.network_path, fighter+historic_network+'_policy_network.h5'))
            self.current_network.dynamic_network.load_weights(os.path.join(self.network_path, fighter+historic_network+'_dynamic_network.h5'))
            self.current_network.reward_network.load_weights(os.path.join(self.network_path, fighter+historic_network+'_reward_network.h5'))
            self.current_network.exploration_network.load_weights(os.path.join(self.network_path, fighter+historic_network+'_exploration_network.h5'))
            print('Found historic network weights, pre-loading MuZero with historic weights.')
        except:
            print('No previous weights found, initializing model')        
        return self.current_network
        


    def save_network(self, step: int, network: BaseNetwork, fighter : str, current_network : str):
        self._networks[step] = step
        networks_df = pd.read_csv(os.path.join(self.network_path,'current_networks.csv'))
        networks_df[fighter+'_current_network'] = int(current_network)
        networks_df['training_loop'] = self.training_loop
        networks_df.to_csv(os.path.join(self.network_path,'current_networks.csv'), index=False)
        network.representation_network.save(os.path.join(self.network_path, fighter+current_network+'_representation_network.h5'))
        network.value_network.save(os.path.join(self.network_path, fighter+current_network+'_value_network.h5'))
        network.policy_network.save(os.path.join(self.network_path, fighter+current_network+'_policy_network.h5'))
        network.dynamic_network.save(os.path.join(self.network_path, fighter+current_network+'_dynamic_network.h5'))
        network.reward_network.save(os.path.join(self.network_path, fighter+current_network+'_reward_network.h5'))
        network.exploration_network.save(os.path.join(self.network_path, fighter+current_network+'_exploration_network.h5'))
        np.save(os.path.join(self.network_path, fighter+current_network+'_optimizer.npy'),self.optimizer.get_weights()) 