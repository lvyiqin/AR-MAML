'''
This file is to gather rollout dataset for meta_dm and meta_pn training. These include databuffer to store transitions/rewards
and transformation function over these transitions (prepared for meta_dm dataloader).
'''

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from physics_simulator.acrobot import acrobot_reward_numpy
from physics_simulator.pendulum import pendulum_reward_numpy


##########################################################################################################################
    # policy modules and rollout api
##########################################################################################################################

    
class RandomPolicy(nn.Module):
    """
    Policy that samples an action uniformly at random from the action space.
    """

    def __init__(self, env, discrete_action_space=True):
        super().__init__()
        self.env = env
        self.discrete_action_space=discrete_action_space

    def forward(self, state):
        #action is regardless of state in random policy
        if self.discrete_action_space:
            action=Variable(torch.tensor(self.env.action_space.sample())) #torch.tensor can directly identify the dtype int or float
        else:
            action=Variable(torch.FloatTensor(self.env.action_space.sample())) #continuous action space uses float dtype
            
        return action


def rollout(env, policy, num_steps):
    """
    Generate one trajectory with System dynamics, return transitions.
    The policy should be random policy and the collected dataset serves the Meta-DM learning.
    """
    cur_state = env.reset()
    
    states = [cur_state]
    actions = []
    
    if env.env_name == 'acrobot':
        rewards = [acrobot_reward_numpy(cur_state)]
    elif env.env_name == 'pendulum':
        rewards = [0.0]
    
    done = False
    t = 0
    while not done:
        t += 1
        # Convert to FloatTensor feedable into a Torch model
        cur_state = torch.FloatTensor(cur_state).unsqueeze(0).cuda() # Ensure input in a batch way
        action = torch.flatten(policy(cur_state))  # Ensure ndims=1
        action = action.data.cpu().numpy() # return action in the form [0], [1] or [2] in acrobot
        next_state, reward, done, _ = env.step(action[0]) # numpy reward function
        
        # Record data
        actions.append(action) 
        states.append(next_state)
        rewards.append(reward)
        
        cur_state = next_state
        
        if t>num_steps :
            done = True
        
    # Convert to numpy arrays
    states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=0),
                                         (states, actions, rewards)))
    # formulate states in shape [num_trans+1,dim_s], actions in shape [num_trans,dim_a], rewards in shape [num_trans+1, ] 
    return states, actions, rewards
    



##########################################################################################################################
    # convert transitions to specific form of dataset for meta dynamics models training
##########################################################################################################################

    
def convert_trajectory_to_training(states, actions, whether_differ=False):
    """
    Convert trajactories to transition dataset -> x as state-action pair and y as difference between states
    
    states-> numpy array of shape [N, state_dim]
    actions-> numpy array of shape [N - 1, action_dim]
    whether_differ-> whether to use the difference between two states as the output
    Returns:
        (x, y): where x is a numpy array of shape [N, state_dim + action_dim] and
        y is a numpy array of shape [N, state_dim]
    """
    if states.ndim == 2: #shape [num_samples, dim]
        assert states.shape[0] == actions.shape[0] + 1
        obs, next_obs = states[:-1], states[1:]
        x = np.concatenate((obs, actions), axis=1) 
        if whether_differ:
            y = next_obs - obs # difference between two states y=s_{t+1}-s_{t}
        else:
            y = next_obs
        
    elif states.ndim ==3: #shape [num_tasks, num_samples, dim]
        assert states.shape[1] == actions.shape[1] + 1
        obs, next_obs = states[:,:-1,:], states[:,1:,:]
        x = np.concatenate((obs, actions), axis=2) 
        if whether_differ:
            y = next_obs - obs # difference between two states y=s_{t+1}-s_{t}
        else:
            y = next_obs
    
    return x, y


def collect_transitions(env, rand_policy, num_steps, whether_reward = False):
    states, actions, rewards = rollout(env, rand_policy, num_steps)
    # print('states', states.ndim) #2
    # print('stat_shape', states.shape) #[202,6]
    
    x,y=convert_trajectory_to_training(states, actions) #return shape [num_samples,dim]
    # print('stat_shape', states.shape, 'actions', actions.shape, 'x', x.shape)
    
    if whether_reward:
        return x, y, rewards
    else:
        return x, y
            

def multi_env_collect_transitions(env_list, num_steps, discrete_action_space):
    x_list, y_list = [], []
    
    rand_policy = RandomPolicy(env_list[0], discrete_action_space)
    # print('len', len(env_list)

    for i in range(len(env_list)):
        # while True:
        #     x, y = collect_transitions(env_list[i], rand_policy, num_steps)
        #     if x.shape == (201,7):
        #         break
        x,y = collect_transitions(env_list[i], rand_policy, num_steps)
        # print('x:', x.shape)
        x_list.append(x)
        y_list.append(y)

    try:
        x_array, y_array = np.stack(x_list), np.array(y_list)
    except ValueError as e:
        print("ValueError occurred. Printing x_list:")
        print([x.shape for x in x_list])
        raise e  
    # x_array, y_array = np.stack(x_list), np.array(y_list)
    x_array, y_array = np.transpose(x_array,(1,0,2)), np.transpose(y_array,(1,0,2)) # return shape [num_samples, num_tasks, dim]
    
    return x_array, y_array



##########################################################################################################################
    # Data buffer for converted transitions in training dynamics models
##########################################################################################################################


class DynamicsDataBuffer(data.Dataset):
    def __init__(self, capacity=1000):
        self.data = []
        self.capacity = capacity
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    
    def __getallitem__(self):
        x_list, y_list = [], []
        allitem = self.data
        for i in range(len(allitem)):
            x_item, y_item = allitem[i]
            x_list.append(x_item)
            y_list.append(y_item)
        x_arr, y_arr = np.array(x_list), np.array(y_list)
        
        return torch.FloatTensor(x_arr), torch.FloatTensor(y_arr)
    

    def push(self, x, y):
        # x/y are of array type
        if x.ndim == 1:
            # In case this is a single datapoint, ensure ndims == 2 (add batch dimension)
            assert y.ndim == 1
            x = x[None, :]
            y = y[None, :]
            
        for i in range(x.shape[0]):
            self.data.append((x[i], y[i]))
            
        # Ensure capacity isn't exceeded
        if len(self.data) > self.capacity:
            del self.data[:len(self.data) - self.capacity]
            
        



def meta_dm_buffer(env_list, num_steps, num_traj, discrete_action_space, whether_buffer=False):
    # Collect transitions for dm learning -> just for meta_dm in training
    dm_buffer = DynamicsDataBuffer(capacity=num_steps*num_traj)
    
    for _ in range(num_traj):
        x_array, y_array = multi_env_collect_transitions(env_list, num_steps, discrete_action_space)
        dm_buffer.push(x_array, y_array) #buffer data shape [num_samples, num_tasks, dim]
        
    x_all, y_all = dm_buffer.__getallitem__() # note that both obs and delta are preprocessed already

    x_all_permute, y_all_permute = x_all.permute(1, 0, 2).contiguous().cuda(), \
        y_all.permute(1,0,2).contiguous().cuda() #return shape [num_tasks, num_samples, dim]
  
    if whether_buffer:
        return dm_buffer
    else:
        return x_all_permute, y_all_permute
