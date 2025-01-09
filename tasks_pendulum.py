import numpy as np
import torch
import random
import math

from physics_simulator.pendulum import sample_batch_envs
from physics_simulator.data_collection import meta_dm_buffer



class PendulumDataset:
    '''
    System identification task in Pendulum Systems.
    '''
    
    def __init__(self):
        # input/output dimensions in regression
        self.num_inputs = 4
        self.num_outputs = 3
        
        # m1 and m2 pendulum mass ranges
        self.m_range = [0.4, 1.6]
        self.l_range = [0.4, 1.6]
        
        # length of the trajectory
        self.num_steps = 200
        
    
    def sample_tasks(self, num_tasks, num_points, init_dis):
        # sample hparams from uniform dist to generate tasks
        sqrt_num_tasks = int(math.sqrt(num_tasks))
        if init_dis == 'Uniform':
            m = np.random.uniform(self.m_range[0], self.m_range[1], sqrt_num_tasks)
            l = np.random.uniform(self.l_range[0], self.l_range[1], sqrt_num_tasks)
        elif init_dis == 'Normal':
            m = np.random.normal(1.0, 0.2, sqrt_num_tasks) 
            l = np.random.normal(1.0, 0.2, sqrt_num_tasks) 
            # clip the sampled initial params into a regular range
            m = np.clip(m, self.m_range[0], self.m_range[1])
            l = np.clip(l, self.l_range[0], self.l_range[1])
        
        ml = torch.zeros(num_tasks, 2)
        k = 0
        for i in range(sqrt_num_tasks):
            for j in range(sqrt_num_tasks):
                ml[k, 0] = m[i]
                ml[k, 1] = l[j]
                k += 1

        x, y = self.sample_datapoints(num_tasks, num_points, ml[:, 0], ml[:, 1])
        
        return x, y     

    
    def sample_init_param(self, num_tasks, init_dis):
        sqrt_num_tasks = int(math.sqrt(num_tasks))  
        # sample batch of hyper-params of tasks for the initialization of tasks
        if init_dis == 'Uniform':
            m_list = np.random.uniform(self.m_range[0], self.m_range[1], sqrt_num_tasks)
            l_list = np.random.uniform(self.l_range[0], self.l_range[1], sqrt_num_tasks)
        elif init_dis == 'Normal':
            m_list = np.random.normal(1.0, 0.2, sqrt_num_tasks)
            l_list = np.random.normal(1.0, 0.2, sqrt_num_tasks)
            m_list = np.clip(m_list, self.m_range[0], self.m_range[1])
            l_list = np.clip(l_list, self.l_range[0], self.l_range[1])
        elif init_dis == 'TruncNorm':
            pass
          
        # init_param_tensor = torch.zeros(sqrt_num_tasks, 2)
        
        # for i in range(sqrt_num_tasks):
        #     init_param_tensor[i, 0] = m_list[i]
        #     init_param_tensor[i, 1] = l_list[i]
        
        init_param_tensor = torch.zeros(num_tasks, 2)
        k = 0
        for i in range(sqrt_num_tasks):
            for j in range(sqrt_num_tasks):
                init_param_tensor[k, 0] = m_list[i]
                init_param_tensor[k, 1] = l_list[j]
                k += 1
        
        return init_param_tensor    
    
    
    def generate_tasks(self, batch_size, num_points, trasformed_param):
        # generate tasks by configuring the transfermed param.
        m_list = trasformed_param[:, 0]
        l_list = trasformed_param[:, 1]
        x, y = self.sample_datapoints(batch_size, num_points, m_list, l_list)
        
        return x, y    
    
    
    def sample_datapoints(self, batch_size, num_points, m_list, l_list):
        # sample batch of tasks    
        env_list = sample_batch_envs(m_list, l_list)
        # collect transitions as dataset for tasks
        inputs, outputs = meta_dm_buffer(env_list,num_steps=num_points,num_traj=1,discrete_action_space=False)

        return inputs, outputs        
    
    
    def sample_meta_dataset(self, batch_x, batch_y, update_batch_size):
        # update_batch_size denotes the number of context points for fast adaptation
        b_size = batch_x.size()[1] # the number of all data points for each task
        idx = random.sample(range(0, b_size), update_batch_size)
        idx = np.array(idx) 
        full_idx = np.arange(b_size)
        idx_t = torch.tensor(list(set(full_idx)-set(idx)),dtype=torch.long)
        idx = torch.tensor(idx)

        idx, idx_t = idx.cuda(), idx_t.cuda()
            
        inputa = torch.index_select(batch_x, dim=1, index=idx)
        labela = torch.index_select(batch_y, dim=1, index=idx) 
        inputb = torch.index_select(batch_x, dim=1, index=idx_t) # b used for testing
        labelb = torch.index_select(batch_y, dim=1, index=idx_t)

        return inputa, labela, inputb, labelb
    
 


    
    
    
    
         
         
