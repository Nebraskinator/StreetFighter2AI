from typing import List
import os
import numpy as np
import cv2
from game.game import Action, AbstractGame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import _pickle as cPickle
import time

class SuperMarioBros(AbstractGame):
    """The Gym CartPole environment"""

    def __init__(self, worlds, discount: float, memory_path: str):
        super().__init__(discount)
        self.world = np.random.choice(worlds) # Pick a random level from the list
        self.env = gym_super_mario_bros.make('SuperMarioBros-'+self.world+'-v1') # Create environment for chosen level
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT) # Limit the action space
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n))) # Legal actions
        self.obs_shape = (96,96,3) # Shape of frame output after resizing
        self.initial_obs = self.env.reset() 
        # Resize initial observation
        self.initial_obs = np.transpose(np.array([cv2.resize(i, (self.obs_shape[1],self.obs_shape[0]), interpolation=cv2.INTER_AREA) for i in np.transpose(self.initial_obs, (2,0,1))]),(1,2,0))
        # Stack a pair of observations to make a full frame
        self.observations = [np.dstack((self.initial_obs,self.initial_obs,))]
        self.steps = 4 # Number of steps to repeat each action
        self.done = False
        self.render_steps = False
        self.prev_obs = 9 # Number of previous observations to include when calling make_image
        self.memory_path = memory_path # Where to save the game
        self.time = 300 # Initial time for reward calculation
        self.x_pos = 40 # Initial position for reward calculation
        self.life = 2 # Initial lives for reward calculation
        self.score = 0 # Initial score for reward calculation
        self.coins = 0 # Initial coins for reward calculation


    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action, render=False) -> int:
        """Execute one step of the game conditioned by the given action."""
        # Create an array to hold all of the observations
        obs = np.zeros((*self.env.observation_space.shape[:2],self.env.observation_space.shape[2]*int(self.steps/2))).astype('uint8')
        # Execute {steps} frames and accumulate the reward
        # We will repeat actions for self.steps frames
        # Pairs of frames will be fused into one frame
        # These fused frames will be stacked to form a sequence
        # that spans self.steps 
        for i in range(self.steps//2): 
            frame = np.zeros((2,*self.env.observation_space.shape)) # array to hold our frame pairs
            for ii in range(2):
                state_frame, reward, done, info = self.env.step(action.index) # send action input to environment
                frame[ii,:,:,:] = state_frame # stack the observation pairs
                if done:
                    break
            # save the max of the frame pair
            obs[:,:,i*int(self.obs_shape[2]):i*int(self.obs_shape[2])+int(self.obs_shape[2])] = np.amax(frame, axis=0)
            if done:
                break
        # Resize the observations
        obs = np.transpose(np.array([cv2.resize(i, (self.obs_shape[1],self.obs_shape[0]), interpolation=cv2.INTER_AREA) for i in np.transpose(obs, (2,0,1))]),(1,2,0))
        self.observations += [obs] # add the observations to the game history
        self.done = done
        # Calculate the reward
        action_reward = np.clip(info['x_pos'] - self.x_pos, -15, 15)
        action_reward += info['time'] - self.time
        action_reward += (info['life'] - self.life) * 15
        action_reward += info['coins'] - self.coins
        if info['flag_get'] == True:
            action_reward += 50
        # update game state values
        self.x_pos = info['x_pos']
        self.time = info['time']
        self.life = info['life']
        self.coins = info['coins']
        return action_reward 

    def close(self):
        self.env.close()

    def terminal(self) -> bool:
        """Is the game is finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        # This function makes a stack of image frames based on the number of previous observations
        # and the shape of representation network input layer.
        # Between each observation frame is a single plane encoding the action
        # in a repeating 5x5 tile
        if state_index == -1:
            state_index = len(self.observations) - 1
        planes_per_obs = self.obs_shape[2]*int(self.steps/2) + 1
        state = np.zeros((*self.obs_shape[:2], (self.prev_obs + 1) * planes_per_obs - 1)).astype('uint8')
        prev_obs_idx = np.arange(state_index - self.prev_obs, state_index)
        if state_index > 0:
            for i in range(self.prev_obs):
                if prev_obs_idx[i] < 0:
                    continue
                else:
                    state[:,:,planes_per_obs*i:planes_per_obs*i+planes_per_obs - 1] = self.observations[prev_obs_idx[i]] 
                    action_pattern_base_shape = int(np.floor(len(self.actions)**0.5) + 1)
                    action_pattern = np.zeros((action_pattern_base_shape**2))
                    action_pattern[:self.history[prev_obs_idx[i]].index] = 255
                    action_pattern = action_pattern.reshape((action_pattern_base_shape,action_pattern_base_shape))
                    action_pattern = np.tile(action_pattern, ((state.shape[0]//action_pattern_base_shape,state.shape[1]//action_pattern_base_shape)))
                    action_pattern = np.pad(action_pattern, ((0,state.shape[0] % action_pattern_base_shape),(0,state.shape[1] % action_pattern_base_shape)))
                    state[:,:,planes_per_obs*i+planes_per_obs - 1] = action_pattern
        state[:,:,-self.obs_shape[2]*int(self.steps/2):] = self.observations[state_index] 
        return state[:,:,-64:].astype('float32') / 255
    
    def save_gif(self,image_name: str):
        # Function that saves a gif to file to replay the game play
        from PIL import Image
        game = np.dstack([i for i in self.observations])
        gif = []
        for i in range(game.shape[2]//3):
            gif += [Image.fromarray(game[:,:,i*3:i*3+3],'RGB')]
            
        gif[0].save(os.path.join(self.memory_path+'Gifs',image_name+'.gif'),
                       save_all=True,
                       append_images=gif[1:],
                       duration=50,
                       loop=0)
    def save_game_to_file(self, game_name):
        # Function that pickles the game and saves it to file so it can be loaded by the trainer agent
        with open(os.path.join(self.memory_path,game_name), 'wb') as game_file:
            cPickle.dump(self, game_file) 

        
