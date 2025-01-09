"""
Neural network models for the regression experiments with adaptively robust maml.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


import normflows as nf
from normflows.flows import Planar, Radial

##########################################################################################################################
    # This part is to introduce the flow module to transform random variables. --> Distribution Adversary
##########################################################################################################################


class Distribution_Adversary(nn.Module):
    def __init__(self, 
                 q0,  
                 latent_size,
                 num_latent_layers,
                 flow_type,
                 hyper_range, 
                 device
                 ):
        '''
        q0: base distribution of task parameters
        latent_size: the dimension of the latent variable
        num_latent_layers: number of layers in NFs
        flow_type: types of NFs
        hyper_range: range of task hyper-parameters, tensor shape [dim_z, 2] e.g. [[4.9, 0.1], [3.0, -1.0]] -> [param_range, range_min]
        device: 'cuda' or 'cpu'
        '''
        
        super(Distribution_Adversary, self).__init__()
        
        self.q0 = q0 
        self.latent_size = latent_size 
        self.num_latent_layers = num_latent_layers 
        self.flow_type = flow_type 
        self.hyper_range = hyper_range
        self.device = device
        
        if flow_type == 'Planar_Flow':
            flows = [Planar(self.latent_size) for k in range(self.num_latent_layers)]
        elif flow_type == 'Radial_Flow':
            flows = [Radial(self.latent_size) for k in range(self.num_latent_layers)]
            
        self.nfm = nf.NormalizingFlow(q0=self.q0, flows=flows)
        self.nfm.to(device)
           

    def forward(self, x, train=True):
        log_det = self.q0.log_prob(x)
        
        z, log_det_forward = self.nfm.forward_and_log_det(x)
    
        min_values = torch.min(z, dim=0).values
        max_values = torch.max(z, dim=0).values
        normalized_data = (z - min_values) / (max_values - min_values)
        
        normalize_tensor = (self.hyper_range.to(self.device)).expand(z.size()[0], -1, -1) # output shape [task_batch, dim_z, 2]
        norm_z = normalize_tensor[:,:,0] * normalized_data + normalize_tensor[:,:,1] # normalize the transformed task into valid ranges
        
        if train:
            log_det_norm = torch.sum(torch.log(normalize_tensor[:,:,0]/ (max_values - min_values)), dim=-1)

            z_reverse, loss = self.nfm.forward_kld(x)
            
            a, b = self.hyper_range[0][1], torch.sum(self.hyper_range[0])
            c, d = self.hyper_range[1][1], torch.sum(self.hyper_range[1])
            condition = (z_reverse[:, 0] >= a) & (z_reverse[:, 0] <= b) & (z_reverse[:, 1] >= c) & (z_reverse[:, 1] <= d)
            z_reverse = z_reverse[condition]
            
            if z_reverse.shape[0] == 0:
                log_det_reverse_total = torch.tensor(0.)
            else:
                log_det_z = self.q0.log_prob(z_reverse)
                log_det_reverse_total = -torch.mean(log_det_z) + loss
            
            return z, norm_z, log_det, log_det_forward, log_det_norm, log_det_reverse_total
        else:
            return norm_z
    
    
    
##########################################################################################################################
    # This part is to introduce the MLP for the implementation of MAML. --> Meta Player
##########################################################################################################################


class Meta_Player(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_weights,
                 task_type,
                 device
                 ):
        '''
        n_inputs: the number of inputs to the network,
        n_outputs: the number of outputs of the network,
        n_weights: for each hidden layer the number of weights, e.g., [128,128,128]
        device: device to deploy, cpu or cuda
        '''
        
        super(Meta_Player, self).__init__()

        # initialise lists for biases and fully connected layers
        self.weights = []
        self.biases = []

        # add one
        if task_type == 'sine':
            self.nodes_per_layer = n_weights + [n_outputs]
        elif task_type == 'acrobot':
            self.nodes_per_layer = n_weights + [n_outputs-2]
        elif task_type == 'pendulum':
            self.nodes_per_layer = n_weights + [n_outputs-1]

        # additional biases
        self.task_context = torch.zeros(0).to(device)
        self.task_context.requires_grad = True

        # set up the shared parts of the layers
        prev_n_weight = n_inputs
        for i in range(len(self.nodes_per_layer)):
            w = torch.Tensor(size=(prev_n_weight, self.nodes_per_layer[i])).to(device)
            w.requires_grad = True
            self.weights.append(w)
            b = torch.Tensor(size=[self.nodes_per_layer[i]]).to(device)
            b.requires_grad = True
            self.biases.append(b)
            prev_n_weight = self.nodes_per_layer[i]

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(len(self.nodes_per_layer)):
            stdv = 1. / math.sqrt(self.nodes_per_layer[i])
            self.weights[i].data.uniform_(-stdv, stdv)
            self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, x, task_type='sine'):
        x = torch.cat((x, self.task_context))

        for i in range(len(self.weights) - 1):
            x = F.relu(F.linear(x, self.weights[i].t(), self.biases[i]))
        
        if task_type == 'sine':
            y = F.linear(x, self.weights[-1].t(), self.biases[-1])
        elif task_type == 'acrobot':
            y = F.linear(x, self.weights[-1].t(), self.biases[-1])
            y = torch.cat((torch.cos(y[...,0:1]),torch.sin(y[...,0:1]),torch.cos(y[...,1:2]),torch.sin(y[...,1:2]),y[...,:2]),dim=-1)
        elif task_type == 'pendulum':
            y = F.linear(x, self.weights[-1].t(), self.biases[-1])
            y = torch.cat((torch.cos(y[...,0:1]),torch.sin(y[...,0:1]),y[...,:1]),dim=-1)            
        
        return y



