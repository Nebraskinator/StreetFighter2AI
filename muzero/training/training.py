"""Training module: this is where MuZero neurons are trained."""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.losses import MSE
from queue import Queue
from config import MuZeroConfig
from networks.network import BaseNetwork
from networks.shared_storage import SharedStorage
from training.replay_buffer import ReplayBuffer
import threading
import time


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, epochs: int):
    network = storage.latest_network()
    optimizer = storage.optimizer
    ex_optimizer = storage.ex_optimizer
    replay_buffer.update_buffer()
    def batch_queue(replay_buffer, q, loops, config):
        for i in range(loops//4):
            q.put(replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps))
    q = Queue(maxsize=7)
    dataloader1 = threading.Thread(target=batch_queue, args=(replay_buffer, q, epochs, config,))
    dataloader1.start()
    
    dataloader2 = threading.Thread(target=batch_queue, args=(replay_buffer, q, epochs, config,))
    dataloader2.start()
    dataloader3 = threading.Thread(target=batch_queue, args=(replay_buffer, q, epochs, config,))
    dataloader3.start()
    dataloader4 = threading.Thread(target=batch_queue, args=(replay_buffer, q, epochs, config,))
    dataloader4.start()

    for _ in range(epochs):
        batch, idx_info, weight_batch = q.get()
        priorities = update_weights(optimizer, ex_optimizer, network, batch, weight_batch)
        replay_buffer.update_priorities(priorities, idx_info)
    dataloader1.join()
    
    dataloader2.join()
    dataloader3.join()
    dataloader4.join()

    storage.save_network(network.training_steps, network)
    time.sleep(1)


def update_weights(optimizer: tf.keras.optimizers, ex_optimizer: tf.keras.optimizers, network: BaseNetwork, batch, weight_batch):
    def scale_gradient(tensor, scale: float):
        """Trick function to scale the gradient in tensorflow"""
        return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor
    priorities = np.zeros((len(batch[0]),len(batch[2])+1))

    def loss():
        loss = 0
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch = batch

        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch, exploration_batch = network.initial_model(np.array(image_batch))
        predictions = []
        batch_size = len(value_batch)
        for j in range(batch_size):
            predictions.append(network._value_transform(value_batch[j]))
        # Only update the element with a policy target
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)

        policy_softmax_batch = tf.nn.softmax(policy_batch)
        priorities[:,0] = np.abs(np.array(predictions) - np.array(target_value_batch)) + 0.1
        # Compute the loss of the first pass
        loss += tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size)*weight_batch)
        loss += tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)*weight_batch)
        loss -= 0.0004 * tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=exploration_batch, labels=policy_softmax_batch)*weight_batch)
        priority_idx = 1
        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch in zip(actions_time_batch, targets_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # Only execute BPTT for elements with an action
            # Creating conditioned_representation: concatenate representations with actions batch
            ## actions_batch = tf.one_hot(actions_batch, network.action_size)
            # Recurrent step from conditioned representation: recurrent + prediction networks
            actions_slice = np.zeros((representation_batch.shape[0],representation_batch.shape[1],representation_batch.shape[2],network.action_size))
            for i in range(representation_batch.shape[0]):
                actions_slice[i,:,:,actions_batch[i]] = 1.
            actions_slice = tf.convert_to_tensor(actions_slice,dtype=tf.float32)
            conditioned_representation_batch = tf.concat((representation_batch, actions_slice), axis=-1)
            representation_batch, reward_batch, value_batch, policy_batch, exploration_batch = network.recurrent_model(
                conditioned_representation_batch)
            policy_softmax_batch = tf.nn.softmax(policy_batch)
            predictions = []
            for jj in range(batch_size):
                predictions.append(network._value_transform(value_batch[jj]))

            priorities[:,priority_idx] = np.abs(np.array(predictions) - np.array(target_value_batch)) + 0.1  
            priority_idx += 1   
            # Only execute BPTT for elements with a policy target

            # Compute the partial loss
            l = (tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size)*weight_batch) +
                 tf.math.reduce_mean(loss_value(target_reward_batch, reward_batch, network.value_support_size)*weight_batch) +
                 tf.math.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)*weight_batch) - 
                 0.0004 * tf.math.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=exploration_batch, labels=policy_softmax_batch)*weight_batch))

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)
        print('loss: '+str(loss))
        return loss
    
    def exploration_loss():
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch = batch

        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch, exploration_batch = network.initial_model(np.array(image_batch))
        policy_softmax_batch = tf.nn.softmax(policy_batch)
        loss = tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=exploration_batch, labels=policy_softmax_batch)*weight_batch)        
        for actions_batch, targets_batch in zip(actions_time_batch, targets_time_batch):

            # Only execute BPTT for elements with an action
            # Creating conditioned_representation: concatenate representations with actions batch
            ## actions_batch = tf.one_hot(actions_batch, network.action_size)
            # Recurrent step from conditioned representation: recurrent + prediction networks
            actions_slice = np.zeros((representation_batch.shape[0],representation_batch.shape[1],representation_batch.shape[2],network.action_size))
            for i in range(representation_batch.shape[0]):
                actions_slice[i,:,:,actions_batch[i]] = 1.
            actions_slice = tf.convert_to_tensor(actions_slice,dtype=tf.float32)
            conditioned_representation_batch = tf.concat((representation_batch, actions_slice), axis=-1)
            representation_batch, reward_batch, value_batch, policy_batch, exploration_batch = network.recurrent_model(
                conditioned_representation_batch)
            policy_softmax_batch = tf.nn.softmax(policy_batch)

            # Compute the partial loss
            l = tf.math.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=exploration_batch, labels=policy_softmax_batch)*weight_batch)

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)
        print('exploration loss: '+str(loss))
        return loss
    
    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    ex_optimizer.minimize(loss=exploration_loss, var_list=network.ex_get_variables())
    print('priorities mean: '+str(np.mean(priorities))+'   priorities max: '+str(np.amax(priorities)))
    network.training_steps += 1
    return priorities

def loss_value(target_value_batch, value_batch, value_support_size: int):
    batch_size = len(target_value_batch)
    targets = np.zeros((batch_size, value_support_size))
    target_value_batch = np.nan_to_num(target_value_batch)
    target_value_batch = target_value_batch
    n_value = np.clip(target_value_batch,0,value_support_size-1.0001)
    floor_value = np.floor(n_value).astype(int)
    rest = n_value - floor_value
    targets[range(batch_size), floor_value.astype(int)] = 1 - rest
    targets[range(batch_size), floor_value.astype(int) + 1] = rest

    return tf.nn.softmax_cross_entropy_with_logits(logits=value_batch, labels=targets)
