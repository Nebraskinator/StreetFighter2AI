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
        self.world = np.random.choice(worlds)
        self.env = gym_super_mario_bros.make('SuperMarioBros-'+self.world+'-v1')
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.obs_shape = (96,96,3)
        self.initial_obs = self.env.reset()
        self.initial_obs = np.transpose(np.array([cv2.resize(i, (self.obs_shape[1],self.obs_shape[0]), interpolation=cv2.INTER_AREA) for i in np.transpose(self.initial_obs, (2,0,1))]),(1,2,0))
        self.observations = [np.dstack((self.initial_obs,self.initial_obs,))]
        self.steps = 4
        self.done = False
        self.render_steps = False
        self.prev_obs = 9
        self.memory_path = memory_path
        self.time = 300
        self.x_pos = 40
        self.life = 2
        self.score = 0
        self.coins = 0


    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action, render=False) -> int:
        """Execute one step of the game conditioned by the given action."""
        obs = np.zeros((*self.env.observation_space.shape[:2],self.env.observation_space.shape[2]*int(self.steps/2))).astype('uint8')
        

        # Execute {steps} frames and accumulate the reward
        for i in range(self.steps//2):
            frame = np.zeros((2,*self.env.observation_space.shape))
            for ii in range(2):
                state_frame, reward, done, info = self.env.step(action.index)
                frame[ii,:,:,:] = state_frame
                if done:
                    break
            obs[:,:,i*int(self.obs_shape[2]):i*int(self.obs_shape[2])+int(self.obs_shape[2])] = np.amax(frame, axis=0)
            if done:
                break
        obs = np.transpose(np.array([cv2.resize(i, (self.obs_shape[1],self.obs_shape[0]), interpolation=cv2.INTER_AREA) for i in np.transpose(obs, (2,0,1))]),(1,2,0))
        self.observations += [obs]
        self.done = done
        action_reward = np.clip(info['x_pos'] - self.x_pos, -15, 15)
        action_reward += info['time'] - self.time
        action_reward += (info['life'] - self.life) * 15
        action_reward += info['coins'] - self.coins
        if info['flag_get'] == True:
            action_reward += 50

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
        with open(os.path.join(self.memory_path,game_name), 'wb') as game_file:
            cPickle.dump(self, game_file) 

        