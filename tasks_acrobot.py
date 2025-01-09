import numpy as np
import torch
import random
import math

from physics_simulator.acrobot import sample_batch_envs
from physics_simulator.data_collection import meta_dm_buffer



class AcrobotDataset:
    '''
    System identification task in Acrobot Systems.
    '''
    
    def __init__(self):
        # input/output dimensions in regression
        self.num_inputs = 7
        self.num_outputs = 6
        
        # m1 and m2 pendulum mass ranges
        self.m_1_range = [0.4, 1.6]
        self.m_2_range = [0.4, 1.6]
        
        # length of the trajectory
        self.num_steps = 200
        
    
    def sample_tasks(self, num_tasks, num_points, init_dis):
        # sample params from uniform dist to generate tasks
        sqrt_num_tasks = int(math.sqrt(num_tasks))
        if init_dis == 'Uniform':
            m_1 = np.random.uniform(self.m_1_range[0], self.m_1_range[1], sqrt_num_tasks)
            m_2 = np.random.uniform(self.m_2_range[0], self.m_2_range[1], sqrt_num_tasks)
        elif init_dis == 'Normal':
            m_1 = np.random.normal(1.0, 0.2, sqrt_num_tasks) 
            m_2 = np.random.normal(1.0, 0.2, sqrt_num_tasks) 
            # clip the sampled initial params into a regular range
            m_1 = np.clip(m_1, self.m_1_range[0], self.m_1_range[1])
            m_2 = np.clip(m_2, self.m_2_range[0], self.m_2_range[1])
        
        m = torch.zeros(num_tasks, 2)
        k = 0
        for i in range(sqrt_num_tasks):
            for j in range(sqrt_num_tasks):
                m[k, 0] = m_1[i]
                m[k, 1] = m_2[j]
                k += 1

        x, y = self.sample_datapoints(num_tasks, num_points, m[:, 0], m[:, 1])
        
        return x, y     

    
    def sample_init_param(self, num_tasks, init_dis):
        # sample batch of hyper-params of tasks for the initialization of tasks
        sqrt_num_tasks = int(math.sqrt(num_tasks)) 
        if init_dis == 'Uniform':
            m_1_list = np.random.uniform(self.m_1_range[0], self.m_1_range[1], sqrt_num_tasks)
            m_2_list = np.random.uniform(self.m_2_range[0], self.m_2_range[1], sqrt_num_tasks)
        elif init_dis == 'Normal':
            m_1_list = np.random.normal(1.0, 0.2, sqrt_num_tasks)
            m_2_list = np.random.normal(1.0, 0.2, sqrt_num_tasks)
            m_1_list = np.clip(m_1_list, self.m_1_range[0], self.m_1_range[1])
            m_2_list = np.clip(m_2_list, self.m_2_range[0], self.m_2_range[1])
        
        elif init_dis == 'TruncNorm':
            pass
                  
        init_param_tensor = torch.zeros(num_tasks, 2)
        k = 0
        for i in range(sqrt_num_tasks):
            for j in range(sqrt_num_tasks):
                init_param_tensor[k, 0] = m_1_list[i]
                init_param_tensor[k, 1] = m_2_list[j]
                k += 1
        
        return init_param_tensor    
    
    
    def generate_tasks(self, batch_size, num_points, trasformed_param):
        # generate tasks by configuring the transfermed param.
        m_1_list = trasformed_param[:, 0]
        m_2_list = trasformed_param[:, 1]
        x, y = self.sample_datapoints(batch_size, num_points, m_1_list, m_2_list)
        
        return x, y    
    
    
    def sample_datapoints(self, batch_size, num_points, m_1_list, m_2_list):
        # sample batch of tasks    
        env_list = sample_batch_envs(m_1_list, m_2_list)
        # collect transitions as dataset for tasks
        inputs, outputs = meta_dm_buffer(env_list,num_steps=num_points,num_traj=1,discrete_action_space=True)
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
    
 


    
    
    
    
         
         
        