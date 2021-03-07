import math

import numpy as np
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Lambda, GRU, Dense, Conv2D,Flatten,LayerNormalization, BatchNormalization,AveragePooling2D, Add, GlobalAveragePooling2D, Activation, LeakyReLU
from tensorflow.python.keras.models import Sequential, Model, Input
import tensorflow as tf
from game.game import Action
from networks.network import BaseNetwork

class SuperMarioBrosNetwork(BaseNetwork):

    def __init__(self,
                 state_size: tuple,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = max_value + 1

        regularizer = regularizers.l2(weight_decay)
        
        def identity(inputs,representation_size):
            layer = Conv2D(representation_size, 3,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = Conv2D(representation_size, 3,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer
 
        def minmax_layer(inputs):
            return (inputs - tf.reduce_min(inputs, [1,2,3],keepdims=True)) / (tf.reduce_max(inputs, [1,2,3],keepdims=True) - tf.reduce_min(inputs, [1,2,3],keepdims=True))
        
        rep_input = Input(self.state_size)
        rep_layer = Conv2D(representation_size//2, 5, padding='same', strides=2, kernel_regularizer=regularizer)(rep_input)
        rep_layer = LayerNormalization(scale=False,trainable=True)(rep_layer)
        rep_layer = LeakyReLU()(rep_layer)   
        rep_layer = identity(rep_layer,representation_size//2)
        rep_layer = Conv2D(representation_size//2, 5, padding='same', strides=2, kernel_regularizer=regularizer)(rep_layer)
        rep_layer = LayerNormalization(scale=False,trainable=True)(rep_layer)
        rep_layer = LeakyReLU()(rep_layer) 
        rep_layer = identity(rep_layer,representation_size//2) 
        rep_layer = identity(rep_layer,representation_size//2) 
        rep_layer = Conv2D(representation_size, 5, padding='same', strides=2, kernel_regularizer=regularizer)(rep_layer)
        rep_layer = LayerNormalization(scale=False,trainable=True)(rep_layer)
        rep_layer = LeakyReLU()(rep_layer)
        rep_layer = identity(rep_layer,representation_size)
        rep_layer = identity(rep_layer,representation_size)
        rep_layer = Conv2D(representation_size, 5, padding='same', strides=2, kernel_regularizer=regularizer)(rep_layer)
        rep_layer = LayerNormalization(scale=False,trainable=True)(rep_layer)
        rep_layer = LeakyReLU()(rep_layer)
        rep_layer_1 = identity(rep_layer,representation_size)
        rep_layer_1 = identity(rep_layer_1,representation_size)
        rep_layer_1 = identity(rep_layer_1,representation_size)
        rep_layer_1 = identity(rep_layer_1,representation_size)
        rep_layer_2 = Add()([rep_layer, rep_layer_1])
        rep_layer_2 = identity(rep_layer_2,representation_size)  
        rep_layer_2 = identity(rep_layer_2,representation_size)
        rep_layer_2 = identity(rep_layer_2,representation_size)
        rep_layer_2 = identity(rep_layer_2,representation_size)
        rep_layer_3 = Add()([rep_layer, rep_layer_1, rep_layer_2])
        rep_layer_3 = identity(rep_layer_3,representation_size)
        rep_layer_3 = identity(rep_layer_3,representation_size)
        rep_layer_3 = identity(rep_layer_3,representation_size)
        rep_layer_3 = identity(rep_layer_3,representation_size)
        rep_layer_4 = Add()([rep_layer, rep_layer_1, rep_layer_2, rep_layer_3])
        rep_layer_4 = identity(rep_layer_4,representation_size)
        rep_layer_4 = identity(rep_layer_4,representation_size)
        rep_layer_4 = identity(rep_layer_4,representation_size)      
        rep_layer_4 = identity(rep_layer_4,representation_size)  
        rep_layer_out = Lambda(minmax_layer)(rep_layer_4)

        representation_network = Model(rep_input, rep_layer_out)
        
        value_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size))
        value_layer = identity(value_input,representation_size)
        value_layer = identity(value_input,representation_size)
        value_layer = GlobalAveragePooling2D()(value_layer)
        value_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(value_layer)
        
        value_network = Model(value_input, value_layer)
        
        policy_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size))
        policy_layer = identity(policy_input,representation_size)
        policy_layer = identity(policy_input,representation_size)
        policy_layer = GlobalAveragePooling2D()(policy_layer)
        policy_layer = Dense(action_size, kernel_regularizer=regularizer)(policy_layer)
        
        policy_network = Model(policy_input, policy_layer)
        
        dynamic_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size+action_size))
        dynamic_layer = identity(dynamic_input, representation_size+action_size)
        dynamic_layer = identity(dynamic_input, representation_size+action_size)
        dynamic_layer = Conv2D(representation_size, 1, padding='same', kernel_regularizer=regularizer)(dynamic_layer)
        dynamic_layer = LayerNormalization(scale=False,trainable=True)(dynamic_layer)
        dynamic_layer = LeakyReLU()(dynamic_layer)
        dynamic_layer_1 = identity(dynamic_layer, representation_size)
        dynamic_layer_1 = identity(dynamic_layer_1, representation_size)
        dynamic_layer_1 = identity(dynamic_layer_1, representation_size)
        dynamic_layer_2 = Add()([dynamic_layer, dynamic_layer_1])
        dynamic_layer_2 = identity(dynamic_layer_2, representation_size)
        dynamic_layer_2 = identity(dynamic_layer_2, representation_size)
        dynamic_layer_2 = identity(dynamic_layer_2, representation_size)        
        dynamic_layer_out = Lambda(minmax_layer)(dynamic_layer_2)

        dynamic_network = Model(dynamic_input, dynamic_layer_out)
        
        reward_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size+action_size))
        reward_layer = identity(reward_input,representation_size+action_size)
        reward_layer = GlobalAveragePooling2D()(reward_layer)
        reward_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(reward_layer)
        
        reward_network = Model(reward_input, reward_layer)
        
        exploration_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size))
        exploration_layer = identity(exploration_input,representation_size)  
        exploration_layer = identity(exploration_input,representation_size) 
        exploration_layer = GlobalAveragePooling2D()(exploration_layer)
        exploration_layer = Dense(action_size, kernel_regularizer=regularizer)(exploration_layer)
        
        exploration_network = Model(exploration_input, exploration_layer)
                
        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network, exploration_network)


    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value)

    def _reward_transform(self, reward: np.array) -> float:
        value = self._softmax(reward)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        action_slice = np.zeros((hidden_state.shape[0],hidden_state.shape[1],self.action_size))
        action_slice[:,:,action.index] = 1.
        conditioned_hidden = np.dstack((hidden_state, action_slice))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values, axis=-1))
        return values_exp / np.sum(values_exp, axis = -1)