import tensorflow as tf
import os
from networks.network import BaseNetwork, UniformNetwork, AbstractNetwork


class SharedStorage(object):
    """Save the different versions of the network."""

    def __init__(self, network: BaseNetwork, uniform_network: UniformNetwork, optimizer: tf.keras.optimizers, ex_optimizer: tf.keras.optimizers, network_path: str):
        self._networks = {}
        self._representation_weights = {}
        self._value_weights = {}
        self._policy_weights = {}
        self._dynamic_weights = {}
        self._reward_weights = {}
        self._exploration_weights = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer
        self.ex_optimizer = ex_optimizer
        self.network_path = network_path

    def latest_network(self) -> AbstractNetwork:
        try:
            self.current_network.representation_network.load_weights(os.path.join(self.network_path, 'representation_network.h5'))
            self.current_network.value_network.load_weights(os.path.join(self.network_path, 'value_network.h5'))
            self.current_network.policy_network.load_weights(os.path.join(self.network_path, 'policy_network.h5'))
            self.current_network.dynamic_network.load_weights(os.path.join(self.network_path, 'dynamic_network.h5'))
            self.current_network.reward_network.load_weights(os.path.join(self.network_path, 'reward_network.h5'))
            self.current_network.exploration_network.load_weights(os.path.join(self.network_path, 'exploration_network.h5'))
            print('Found previous network weights, pre-loading MuZero with saved weights.')
        except:
            print('No previous weights found, initializing model')
                            

        return self.current_network        

    def save_network(self, step: int, network: BaseNetwork):
        self._networks[step] = step
        self._representation_weights[step] = network.representation_network.get_weights()
        self._value_weights[step] = network.value_network.get_weights()
        self._policy_weights[step] = network.policy_network.get_weights()
        self._dynamic_weights[step] = network.dynamic_network.get_weights()
        self._reward_weights[step] = network.reward_network.get_weights() 
        self._exploration_weights[step] = network.exploration_network.get_weights()
        network.representation_network.save(os.path.join(self.network_path, 'representation_network.h5'))
        network.value_network.save(os.path.join(self.network_path, 'value_network.h5'))
        network.policy_network.save(os.path.join(self.network_path, 'policy_network.h5'))
        network.dynamic_network.save(os.path.join(self.network_path, 'dynamic_network.h5'))
        network.reward_network.save(os.path.join(self.network_path, 'reward_network.h5'))
        network.exploration_network.save(os.path.join(self.network_path, 'exploration_network.h5'))
