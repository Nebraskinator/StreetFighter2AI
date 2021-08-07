import math

import numpy as np
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Lambda, DepthwiseConv2D, GRU, Dense, Conv2D,Flatten,LayerNormalization, BatchNormalization,AveragePooling2D, Add, GlobalAveragePooling2D, Activation, LeakyReLU
from tensorflow.python.keras.models import Sequential, Model, Input
import tensorflow as tf
from game.game import Action
from networks.network import BaseNetwork
import tensorflow.keras.backend as K

class Involution2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'same', dilation_rate = 1, groups = 1, reduce_ratio = 1):
        super(Involution2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.reduce_mapping = tf.keras.Sequential(
            [
                Conv2D(filters // reduce_ratio, 1, padding = padding), 
                BatchNormalization(), 
                Activation('relu'), 
            ]
        )
        self.span_mapping = Conv2D(kernel_size * kernel_size * groups, 1, padding = padding)
        self.initial_mapping = Conv2D(self.filters, 1, padding = padding)
        if strides > 1:
            self.o_mapping = AveragePooling2D(strides)
    
    def call(self, x):
        weight = self.span_mapping(self.reduce_mapping(x if self.strides == 1 else self.o_mapping(x)))
        _, h, w, c = K.int_shape(weight)
        weight = K.expand_dims(K.reshape(weight, (-1, h, w, self.groups, self.kernel_size * self.kernel_size)), axis = 4)
        out = tf.image.extract_patches(images = x if self.filters == c else self.initial_mapping(x),  
                                       sizes = [1, self.kernel_size, self.kernel_size, 1], 
                                       strides = [1, self.strides, self.strides, 1], 
                                       rates = [1, self.dilation_rate, self.dilation_rate, 1], 
                                       padding = "SAME" if self.padding == 'same' else "VALID")
        out = K.reshape(out, (-1, h, w, self.groups, self.filters // self.groups, self.kernel_size * self.kernel_size))
        out = K.sum(weight * out, axis = -1)
        out = K.reshape(out, (-1, h, w, self.filters))
        return out

class StreetFighter2Network(BaseNetwork):

    def __init__(self,
                 state_size: tuple,
                 action_size: int,
                 representation_size: int,
                 min_value: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-5,
                 representation_activation: str = 'tanh'):
        self.min_value = min_value
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = max_value - min_value + 1
        self.optimizer_weights = None

        regularizer = regularizers.l2(weight_decay)
        
        def conv_identity(inputs,k,input_filters,output_filters):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            layer = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k, padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer

        def conv_reduction(inputs,k,input_filters,output_filters):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(output_filters, k, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer    

        def mb_reduction(inputs,k,input_filters,output_filters,expand):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), strides=2, padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)  
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def mb_identity(inputs,k,input_filters,output_filters,expand):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def minmax_layer(inputs):
            return (inputs - tf.reduce_min(inputs, [1,2,3],keepdims=True)) / (tf.reduce_max(inputs, [1,2,3],keepdims=True) - tf.reduce_min(inputs, [1,2,3],keepdims=True))
        
        rep_input = Input(self.state_size)
        rep_layer = conv_reduction(rep_input,3,self.state_size[2],32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,32,32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,32,32) # 48x48x16
        rep_layer = mb_identity(rep_layer,3,32,32,24*6) # 48x48x24
        rep_layer = mb_identity(rep_layer,3,32,32,24*6) # 48x48x24
        rep_layer = mb_reduction(rep_layer,5,24,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,5,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40        
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_reduction(rep_layer,5,40,80,6*80) # 12x12x80
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80      
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80 
        rep_layer = mb_reduction(rep_layer,5,80,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112        
        
        
#        rep_layer_out = Lambda(minmax_layer)(rep_layer)

        representation_network = Model(rep_input, rep_layer)
        
        value_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        value_layer = mb_identity(value_input,3,112,112,6*112)
        value_layer = mb_identity(value_layer,3,112,112,6*112)
        value_layer_pooled = GlobalAveragePooling2D()(value_layer)
        value_layer_pooled = Dense(self.value_support_size, kernel_regularizer=regularizer)(value_layer_pooled)
        
        value_network = Model(value_input, value_layer_pooled)
        
        policy_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        policy_layer = mb_identity(policy_input,3,112,112,6*112)
        policy_layer = mb_identity(policy_layer,3,112,112,6*112)
        policy_layer = GlobalAveragePooling2D()(policy_layer)
        policy_layer = Dense(action_size, kernel_regularizer=regularizer)(policy_layer)
        
        policy_network = Model(policy_input, policy_layer)
        
        dynamic_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        dynamic_layer = mb_identity(dynamic_input,5,112+action_size,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
#        dynamic_layer_out = Lambda(minmax_layer)(dynamic_layer)

        dynamic_network = Model(dynamic_input, dynamic_layer)
        
        reward_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        reward_layer = mb_identity(reward_input,3,112+action_size,112,6*112)
        reward_layer = mb_identity(reward_layer,3,112,112,6*112)
        reward_layer = GlobalAveragePooling2D()(reward_layer)
        reward_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(reward_layer)
        
        reward_network = Model(reward_input, reward_layer)
        
        exploration_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        exploration_layer = mb_identity(exploration_input,3,112,112,6*112)
        exploration_layer = mb_identity(exploration_layer,3,112,112,6*112)
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
        return np.asscalar(value) + self.min_value

    def _reward_transform(self, reward: np.array) -> float:
        value = self._softmax(reward)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value) + self.min_value

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        action_slice = np.zeros((hidden_state.shape[0],hidden_state.shape[1],self.action_size))
        action_slice[:,:,action.index] = 1.
        conditioned_hidden = np.dstack((hidden_state, action_slice))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values, axis=-1))
        return values_exp / np.sum(values_exp, axis = -1)     

class StreetFighter2Network_v4(BaseNetwork):

    def __init__(self,
                 state_size: tuple,
                 action_size: int,
                 representation_size: int,
                 min_value: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-5,
                 representation_activation: str = 'tanh'):
        self.min_value = min_value
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = max_value - min_value + 1
        self.optimizer_weights = None

        regularizer = regularizers.l2(weight_decay)
        
        def conv_identity(inputs,k,input_filters,output_filters):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            layer = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k, padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer

        def conv_reduction(inputs,k,input_filters,output_filters):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(output_filters, k, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer    

        def mb_reduction(inputs,k,input_filters,output_filters,expand):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), strides=2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)  
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def mb_identity(inputs,k,input_filters,output_filters,expand):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def minmax_layer(inputs):
            return (inputs - tf.reduce_min(inputs, [1,2,3],keepdims=True)) / (tf.reduce_max(inputs, [1,2,3],keepdims=True) - tf.reduce_min(inputs, [1,2,3],keepdims=True))
        
        rep_input = Input(self.state_size)
        rep_layer = conv_reduction(rep_input,3,self.state_size[2],32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,32,32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,32,32) # 48x48x16
        rep_layer = mb_identity(rep_layer,3,32,32,24*6) # 48x48x24
        rep_layer = mb_identity(rep_layer,3,32,32,24*6) # 48x48x24
        rep_layer = mb_reduction(rep_layer,5,24,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,5,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40        
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_reduction(rep_layer,5,40,80,6*80) # 12x12x80
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80      
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80 
        rep_layer = mb_reduction(rep_layer,5,80,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112        
        
        
#        rep_layer_out = Lambda(minmax_layer)(rep_layer)

        representation_network = Model(rep_input, rep_layer)
        
        value_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        value_layer = mb_identity(value_input,3,112,112,6*112)
        value_layer = mb_identity(value_layer,3,112,112,6*112)
        value_layer_pooled = GlobalAveragePooling2D()(value_layer)
        value_layer_pooled = Dense(self.value_support_size, kernel_regularizer=regularizer)(value_layer_pooled)
        
        value_network = Model(value_input, value_layer_pooled)
        
        policy_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        policy_layer = mb_identity(policy_input,3,112,112,6*112)
        policy_layer = mb_identity(policy_layer,3,112,112,6*112)
        policy_layer = GlobalAveragePooling2D()(policy_layer)
        policy_layer = Dense(action_size, kernel_regularizer=regularizer)(policy_layer)
        
        policy_network = Model(policy_input, policy_layer)
        
        dynamic_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        dynamic_layer = mb_identity(dynamic_input,5,112+action_size,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
#        dynamic_layer_out = Lambda(minmax_layer)(dynamic_layer)

        dynamic_network = Model(dynamic_input, dynamic_layer)
        
        reward_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        reward_layer = mb_identity(reward_input,3,112+action_size,112,6*112)
        reward_layer = mb_identity(reward_layer,3,112,112,6*112)
        reward_layer = GlobalAveragePooling2D()(reward_layer)
        reward_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(reward_layer)
        
        reward_network = Model(reward_input, reward_layer)
        
        exploration_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        exploration_layer = mb_identity(exploration_input,3,112,112,6*112)
        exploration_layer = mb_identity(exploration_layer,3,112,112,6*112)
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
        return np.asscalar(value) + self.min_value

    def _reward_transform(self, reward: np.array) -> float:
        value = self._softmax(reward)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value) + self.min_value

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        action_slice = np.zeros((hidden_state.shape[0],hidden_state.shape[1],self.action_size))
        action_slice[:,:,action.index] = 1.
        conditioned_hidden = np.dstack((hidden_state, action_slice))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values, axis=-1))
        return values_exp / np.sum(values_exp, axis = -1) 

class StreetFighter2Network_v3(BaseNetwork):

    def __init__(self,
                 state_size: tuple,
                 action_size: int,
                 representation_size: int,
                 min_value: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-5,
                 representation_activation: str = 'tanh'):
        self.min_value = min_value
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = max_value - min_value + 1
        self.optimizer_weights = None

        regularizer = regularizers.l2(weight_decay)
        
        def conv_identity(inputs,k,input_filters,output_filters):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            layer = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k, padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer

        def conv_reduction(inputs,k,input_filters,output_filters):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(output_filters, k, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer    

        def mb_reduction(inputs,k,input_filters,output_filters,expand):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), strides=2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)  
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def mb_identity(inputs,k,input_filters,output_filters,expand):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def minmax_layer(inputs):
            return (inputs - tf.reduce_min(inputs, [1,2,3],keepdims=True)) / (tf.reduce_max(inputs, [1,2,3],keepdims=True) - tf.reduce_min(inputs, [1,2,3],keepdims=True))
        
        rep_input = Input(self.state_size)
        rep_layer = conv_reduction(rep_input,3,self.state_size[2],32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,32,32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,16,16) # 48x48x16
        rep_layer = mb_identity(rep_layer,3,16,24,24*6) # 48x48x24
        rep_layer = mb_identity(rep_layer,3,24,24,24*6) # 48x48x24
        rep_layer = mb_reduction(rep_layer,5,24,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,5,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40        
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_reduction(rep_layer,5,40,80,6*80) # 12x12x80
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80      
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80 
        rep_layer = mb_reduction(rep_layer,5,80,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112        
        
        
#        rep_layer_out = Lambda(minmax_layer)(rep_layer)

        representation_network = Model(rep_input, rep_layer)
        
        value_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        value_layer = mb_identity(value_input,3,112,112,6*112)
        value_layer = mb_identity(value_layer,3,112,112,6*112)
        value_layer_pooled = GlobalAveragePooling2D()(value_layer)
        value_layer_pooled = Dense(self.value_support_size, kernel_regularizer=regularizer)(value_layer_pooled)
        
        value_network = Model(value_input, value_layer_pooled)
        
        policy_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        policy_layer = mb_identity(policy_input,3,112,112,6*112)
        policy_layer = mb_identity(policy_layer,3,112,112,6*112)
        policy_layer = GlobalAveragePooling2D()(policy_layer)
        policy_layer = Dense(action_size, kernel_regularizer=regularizer)(policy_layer)
        
        policy_network = Model(policy_input, policy_layer)
        
        dynamic_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        dynamic_layer = mb_identity(dynamic_input,5,112+action_size,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
#        dynamic_layer_out = Lambda(minmax_layer)(dynamic_layer)

        dynamic_network = Model(dynamic_input, dynamic_layer)
        
        reward_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        reward_layer = mb_identity(reward_input,3,112+action_size,112,6*112)
        reward_layer = mb_identity(reward_layer,3,112,112,6*112)
        reward_layer = GlobalAveragePooling2D()(reward_layer)
        reward_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(reward_layer)
        
        reward_network = Model(reward_input, reward_layer)
        
        exploration_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        exploration_layer = mb_identity(exploration_input,3,112,112,6*112)
        exploration_layer = mb_identity(exploration_layer,3,112,112,6*112)
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
        return np.asscalar(value) + self.min_value

    def _reward_transform(self, reward: np.array) -> float:
        value = self._softmax(reward)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value) + self.min_value

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        action_slice = np.zeros((hidden_state.shape[0],hidden_state.shape[1],self.action_size))
        action_slice[:,:,action.index] = 1.
        conditioned_hidden = np.dstack((hidden_state, action_slice))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values, axis=-1))
        return values_exp / np.sum(values_exp, axis = -1)        
  
class StreetFighter2Network_v2(BaseNetwork):

    def __init__(self,
                 state_size: tuple,
                 action_size: int,
                 representation_size: int,
                 min_value: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-5,
                 representation_activation: str = 'tanh'):
        self.min_value = min_value
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = max_value - min_value + 1
        self.optimizer_weights = None

        regularizer = regularizers.l2(weight_decay)
        
        def conv_identity(inputs,k,input_filters,output_filters):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            layer = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k, padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer

        def conv_reduction(inputs,k,input_filters,output_filters):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(output_filters, k, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = LeakyReLU()(layer)
            layer = Conv2D(output_filters, k,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer    

        def mb_reduction(inputs,k,input_filters,output_filters,expand):
            skip = Conv2D(output_filters, 1, strides = 2, padding='same', kernel_regularizer=regularizer)(inputs)
            skip = LayerNormalization(scale=False,trainable=True)(skip)
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), strides=2, padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)  
            layer = Add()([skip,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def mb_identity(inputs,k,input_filters,output_filters,expand):
            if input_filters != output_filters:
                inputs = Conv2D(output_filters, 1, padding='same', kernel_regularizer=regularizer)(inputs)
                inputs = LayerNormalization(scale=False,trainable=True)(inputs)  
            
            layer = Conv2D(expand, 1,padding='same', kernel_regularizer=regularizer)(inputs)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)
            layer = DepthwiseConv2D((k,k), padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)
            layer = LeakyReLU()(layer)            
            layer = Conv2D(output_filters, 1,padding='same', kernel_regularizer=regularizer)(layer)
            layer = LayerNormalization(scale=False,trainable=True)(layer)            
            layer = Add()([inputs,layer])
            layer = LeakyReLU()(layer)
            return layer 

        def minmax_layer(inputs):
            return (inputs - tf.reduce_min(inputs, [1,2,3],keepdims=True)) / (tf.reduce_max(inputs, [1,2,3],keepdims=True) - tf.reduce_min(inputs, [1,2,3],keepdims=True))
        
        rep_input = Input(self.state_size)
        rep_layer = conv_reduction(rep_input,3,self.state_size[2],32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,32,32) # 48x48x32
        rep_layer = mb_identity(rep_layer,3,32,16,16) # 48x48x16
        rep_layer = mb_identity(rep_layer,3,16,24,24*6) # 48x48x24
        rep_layer = mb_identity(rep_layer,3,24,24,24*6) # 48x48x24
        rep_layer = mb_reduction(rep_layer,5,24,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,5,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40        
        rep_layer = mb_identity(rep_layer,3,40,40,6*40) # 24x24x40
        rep_layer = mb_reduction(rep_layer,5,40,80,6*80) # 12x12x80
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80      
        rep_layer = mb_identity(rep_layer,5,80,80,6*80) # 12x12x80 
        rep_layer = mb_reduction(rep_layer,5,80,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,5,112,112,6*112) # 6x6x112
        rep_layer = mb_identity(rep_layer,3,112,112,6*112) # 6x6x112
#        rep_layer_out = Lambda(minmax_layer)(rep_layer)

        representation_network = Model(rep_input, rep_layer)
        
        value_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        value_layer = mb_identity(value_input,3,112,112,6*112)
        value_layer = conv_identity(value_layer,3,112,112)
        value_layer_pooled = GlobalAveragePooling2D()(value_layer)
        value_layer_pooled = Dense(self.value_support_size, kernel_regularizer=regularizer)(value_layer_pooled)
        
        value_network = Model(value_input, value_layer_pooled)
        
        policy_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        policy_layer = mb_identity(policy_input,3,112,112,6*112)
        policy_layer = conv_identity(policy_layer,3,112,112)
        policy_layer = GlobalAveragePooling2D()(policy_layer)
        policy_layer = Dense(action_size, kernel_regularizer=regularizer)(policy_layer)
        
        policy_network = Model(policy_input, policy_layer)
        
        dynamic_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        dynamic_layer = mb_identity(dynamic_input,5,112+action_size,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,5,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
        dynamic_layer = mb_identity(dynamic_layer,3,112,112,6*112)
#        dynamic_layer_out = Lambda(minmax_layer)(dynamic_layer)

        dynamic_network = Model(dynamic_input, dynamic_layer)
        
        reward_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112+action_size))
        reward_layer = mb_identity(reward_input,3,112+action_size,112,6*112)
        reward_layer = conv_identity(reward_layer,3,112,112)
        reward_layer = GlobalAveragePooling2D()(reward_layer)
        reward_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(reward_layer)
        
        reward_network = Model(reward_input, reward_layer)
        
        exploration_input = Input((self.state_size[0]//16, self.state_size[1]//16, 112))
        exploration_layer = mb_identity(exploration_input,3,112,112,6*112)
        exploration_layer = conv_identity(exploration_layer,3,112,112)
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
        return np.asscalar(value) + self.min_value

    def _reward_transform(self, reward: np.array) -> float:
        value = self._softmax(reward)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value) + self.min_value

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        action_slice = np.zeros((hidden_state.shape[0],hidden_state.shape[1],self.action_size))
        action_slice[:,:,action.index] = 1.
        conditioned_hidden = np.dstack((hidden_state, action_slice))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values, axis=-1))
        return values_exp / np.sum(values_exp, axis = -1)     
    
class StreetFighter2Network_v1(BaseNetwork):

    def __init__(self,
                 state_size: tuple,
                 action_size: int,
                 representation_size: int,
                 min_value: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.min_value = min_value
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = max_value - min_value + 1
        self.optimizer_weights = None

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
        rep_layer_5 = Add()([rep_layer, rep_layer_1, rep_layer_2, rep_layer_3, rep_layer_4])
        rep_layer_5 = identity(rep_layer_5,representation_size)
        rep_layer_5 = identity(rep_layer_5,representation_size)
        rep_layer_5 = identity(rep_layer_5,representation_size)      
        rep_layer_5 = identity(rep_layer_5,representation_size) 
        rep_layer_out = Lambda(minmax_layer)(rep_layer_5)

        representation_network = Model(rep_input, rep_layer_out)
        
        value_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size))
        value_layer = identity(value_input,representation_size)
        value_layer = identity(value_layer,representation_size)
        value_layer_pooled = GlobalAveragePooling2D()(value_layer)
        value_layer_pooled = Dense(self.value_support_size, kernel_regularizer=regularizer)(value_layer_pooled)
        
        value_network = Model(value_input, value_layer_pooled)
        
        policy_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size))
        policy_layer = identity(policy_input,representation_size)
        policy_layer = identity(policy_layer,representation_size)
        policy_layer = GlobalAveragePooling2D()(policy_layer)
        policy_layer = Dense(action_size, kernel_regularizer=regularizer)(policy_layer)
        
        policy_network = Model(policy_input, policy_layer)
        
        dynamic_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size+action_size))
        dynamic_layer = identity(dynamic_input, representation_size+action_size)
        dynamic_layer = identity(dynamic_layer, representation_size+action_size)
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
        dynamic_layer_3 = Add()([dynamic_layer, dynamic_layer_1, dynamic_layer_2])
        dynamic_layer_3 = identity(dynamic_layer_3, representation_size)
        dynamic_layer_3 = identity(dynamic_layer_3, representation_size)
        dynamic_layer_3 = identity(dynamic_layer_3, representation_size)
        dynamic_layer_out = Lambda(minmax_layer)(dynamic_layer_3)

        dynamic_network = Model(dynamic_input, dynamic_layer_out)
        
        reward_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size+action_size))
        reward_layer = identity(reward_input,representation_size+action_size)
        reward_layer = identity(reward_layer,representation_size+action_size)
        reward_layer = GlobalAveragePooling2D()(reward_layer)
        reward_layer = Dense(self.value_support_size, kernel_regularizer=regularizer)(reward_layer)
        
        reward_network = Model(reward_input, reward_layer)
        
        exploration_input = Input((self.state_size[0]//16, self.state_size[1]//16, representation_size))
        exploration_layer = identity(exploration_input,representation_size)
        exploration_layer = identity(exploration_layer,representation_size)
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
        return np.asscalar(value) + self.min_value

    def _reward_transform(self, reward: np.array) -> float:
        value = self._softmax(reward)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value) + self.min_value

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        action_slice = np.zeros((hidden_state.shape[0],hidden_state.shape[1],self.action_size))
        action_slice[:,:,action.index] = 1.
        conditioned_hidden = np.dstack((hidden_state, action_slice))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values, axis=-1))
        return values_exp / np.sum(values_exp, axis = -1)    