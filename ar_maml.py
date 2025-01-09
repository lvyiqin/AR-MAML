"""
Regression experiment using MAML.
"""

import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim
import math

import utils
import random
import tasks_sine, tasks_acrobot, tasks_pendulum
from logger import Logger
from torch.nn import MSELoss 
from ar_maml_model import Distribution_Adversary, Meta_Player
from normflows.distributions import DiagGaussian, Uniform, GaussianDistribution


def cal_cvar(data):
    alpha = [0.9, 0.7, 0.5]
    cvar = []
    for i in alpha: 
        max_data = sorted(data, reverse=True)[0:int(len(data) * (1-i))]
        cvar.append(np.mean(max_data))
    return cvar


def ar_ml_run(args, 
              log_interval=500, 
              dist_interval=1, 
              rerun=False, 
              game_framework=False):
    
    # assert args.ar_maml

    # correctly seed everything
    utils.set_seed(args.seed)

    # --- initialise everything ---
    
    ###################################################################################
        # to introduce Distribution Adversary --> Sample tasks and transform tasks
    ###################################################################################
    print('game_framework', game_framework)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the task family
    if args.task == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal()
        task_family_valid = tasks_sine.RegressionTasksSinusoidal()
        task_family_test = tasks_sine.RegressionTasksSinusoidal()
        low_bound = torch.tensor([0.1, 0.0]) # Lower bounds for each dimension
        high_bound = torch.tensor([5.0, np.pi])  # Upper bounds for each dimension
        mu = torch.tensor([2.5, 1.5])
        sigma = torch.tensor([0.8, 0.5])
        hyper_range = torch.Tensor([[4.9, 0.1], [np.pi, 0]])
    elif args.task == 'acrobot':
        task_family_train = tasks_acrobot.AcrobotDataset()
        task_family_valid = tasks_acrobot.AcrobotDataset()
        task_family_test = tasks_acrobot.AcrobotDataset()
        low_bound = torch.tensor([0.4, 0.4]) # Lower bounds for each dimension
        high_bound = torch.tensor([1.6, 1.6])  # Upper bounds for each dimension
        mu = torch.tensor([1.0, 1.0])
        sigma = torch.tensor([0.2, 0.2])
        hyper_range = torch.Tensor([[1.2, 0.4], [1.2, 0.4]]) 
    elif args.task == 'pendulum':
        task_family_train = tasks_pendulum.PendulumDataset()
        task_family_valid = tasks_pendulum.PendulumDataset()    
        task_family_test = tasks_pendulum.PendulumDataset()
        low_bound = torch.tensor([0.4, 0.4]) # Lower bounds for each dimension
        high_bound = torch.tensor([1.6, 1.6])  # Upper bounds for each dimension
        mu = torch.tensor([1.0, 1.0])
        sigma = torch.tensor([0.2, 0.2])
        hyper_range = torch.Tensor([[1.2, 0.4], [1.2, 0.4]])         
    else:
        raise NotImplementedError
    
    low_bound = low_bound.to(device)
    high_bound = high_bound.to(device)
    mu = mu.to(device)
    sigma = sigma.to(device)
    hyper_range = hyper_range.to(device)

    # intitialise distribution-optimiser
    if args.init_dis == 'Uniform':
        dist_model = Distribution_Adversary(Uniform(shape=1, low=low_bound, high=high_bound), 
                                            args.latent_size, 
                                            args.num_latent_layers, 
                                            args.flow_type,
                                            hyper_range,
                                            device
                                            ).to(device)
    elif args.init_dis == 'Normal':
        dist_model = Distribution_Adversary(GaussianDistribution(shape=2, loc=mu, scale=sigma), 
                                            args.latent_size, 
                                            args.num_latent_layers, 
                                            args.flow_type,
                                            hyper_range,
                                            device
                                            ).to(device)
    dist_optimiser = optim.Adam(dist_model.parameters(), args.lr_dist)
    dist_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dist_optimiser, T_max=5000, eta_min=0)    

      
    ###################################################################################
        # to introduce Meta Player --> Enable fast adaptation to sampled tasks
    ###################################################################################
    
    # initialise network
    model_inner = Meta_Player(task_family_train.num_inputs,
                              task_family_train.num_outputs,
                              n_weights=args.num_hidden_layers,
                              task_type=args.task,
                              device=device
                              ).to(device)
    model_outer = copy.deepcopy(model_inner)

    # intitialise meta-optimiser
    meta_optimiser = optim.Adam(model_outer.weights + model_outer.biases + [model_outer.task_context],
                                args.lr_meta)
    
    
    ###################################################################################
        # update the Meta Player
    ###################################################################################
    
    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model_outer)
    train_losses_mean = []
    train_losses_max = []
    train_losses_std = []
    train_losses_cvar = []

    dist_losses_mean = []
    dist_losses_max = []
    dist_losses_std = []
    dist_losses_kl = []
    dist_entropy = []
    dist_entropy_forward = []
    dist_entropy_norm = []
    lrs = []

    val_losses_mean = []
    val_losses_max = []
    val_losses_std = []
    val_losses_cvar = []
    
    val_init_losses_mean = []
    val_init_losses_max = []
    val_init_losses_cvar = []

    if args.run_type == 'armaml':
        exp_string_outer = 'armaml_' + 'model_outer_' + args.init_dis + '_tpm_' + str(args.tasks_per_metaupdate) + '_K' + str(args.update_batch_size) + '_n' + str(args.n_iter) + args.flow_type + str(args.num_latent_layers) + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) + '.pt'
        exp_string_dist = 'armaml_' + 'dist_model_' + args.init_dis + '_tpm_' + str(args.tasks_per_metaupdate) + '_K' + str(args.update_batch_size) + '_n' + str(args.n_iter) + args.flow_type + str(args.num_latent_layers) + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) + '.pt'
    elif args.run_type == 'maml':
        exp_string_outer = 'maml_' + 'model_outer_' + args.init_dis + '_tpm_' + str(args.tasks_per_metaupdate) + '_K' + str(args.update_batch_size) + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + '.pt'
    else:
        exp_string_outer = 'drmaml_' + 'model_outer_' + args.init_dis + '_tpm_' + str(args.tasks_per_metaupdate) + '_K' + str(args.update_batch_size) + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + '.pt'

    min_loss = 1.0
    os.makedirs(args.task + '_logs_model/', exist_ok=True)
    for i_iter in range(args.n_iter):

        # copy weights of network
        copy_weights = [w.clone() for w in model_outer.weights]
        copy_biases = [b.clone() for b in model_outer.biases]
        copy_context = model_outer.task_context.clone()

        # get all shared parameters and initialise cumulative gradient
        meta_gradient = [0 for _ in range(len(copy_weights + copy_biases) + 1)]

        # sample tasks
        if game_framework:
            # sample initial hyper params of the tasks
            init_param_tensor = task_family_train.sample_init_param(args.tasks_per_metaupdate, args.init_dis).to(device)
            # generate hyper params after transformation as norm_z
            z, norm_z, log_det, log_det_forward, log_det_norm, log_p_phi = dist_model.forward(init_param_tensor, train=True)
            x_all, y_all = task_family_train.generate_tasks(args.tasks_per_metaupdate, args.task_num_data_points, norm_z.cpu().detach().numpy())
        else:
            x_all, y_all = task_family_train.sample_tasks(args.tasks_per_metaupdate, args.task_num_data_points, args.init_dis)
    
        x_c, y_c, x_t, y_t = task_family_train.sample_meta_dataset(x_all, y_all, args.update_batch_size)
        x_c, y_c, x_t, y_t = x_c.to(device), y_c.to(device), x_t.to(device), y_t.to(device)

        # introduce the normalized fast adaptation loss (detached ones) for the adversary updates 
        norm_batch_mse = torch.zeros(args.tasks_per_metaupdate).to(device)
        loss_batch_mse = []
        for t in range(args.tasks_per_metaupdate):
            # reset network weights
            model_inner.weights = [w.clone() for w in copy_weights]
            model_inner.biases = [b.clone() for b in copy_biases]
            model_inner.task_context = copy_context.clone()

            # get data for current task
            for _ in range(args.num_inner_updates):
                train_inputs, train_targets = x_c[t,:,:], y_c[t,:,:]
                # forward through model
                train_outputs = model_inner(train_inputs, task_type=args.task)
            
                # ------------ update on current task ------------

                # compute loss for current task
                loss_task = F.mse_loss(train_outputs, train_targets)

                # compute the gradient wrt current model
                params = [w for w in model_inner.weights] + [b for b in model_inner.biases] + [model_inner.task_context]
                grads = torch.autograd.grad(loss_task, params, create_graph=True, retain_graph=True)

                # make an update on the inner model using the current model (to build up computation graph)
                for i in range(len(model_inner.weights)):
                    if not args.first_order:
                        model_inner.weights[i] = model_inner.weights[i] - args.lr_inner * grads[i]
                    else:
                        model_inner.weights[i] = model_inner.weights[i] - args.lr_inner * grads[i].detach()
                for j in range(len(model_inner.biases)):
                    if not args.first_order:
                        model_inner.biases[j] = model_inner.biases[j] - args.lr_inner * grads[i + j + 1]
                    else:
                        model_inner.biases[j] = model_inner.biases[j] - args.lr_inner * grads[i + j + 1].detach()
                if not args.first_order:
                    model_inner.task_context = model_inner.task_context - args.lr_inner * grads[i + j + 2]
                else:
                    model_inner.task_context = model_inner.task_context - args.lr_inner * grads[i + j + 2].detach()

            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs, test_targets = x_t[t,:,:], y_t[t,:,:]
            
            # get outputs after update
            test_outputs = model_inner(test_inputs, task_type=args.task)
           
            # compute loss (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets) 
            loss_batch_mse.append(loss_meta.detach().item())

            loss_adapt = MSELoss(reduction='mean')
            task_mse = loss_adapt(test_outputs, test_targets)
            norm_batch_mse[t] = task_mse
            
            # compute gradient w.r.t. *outer model*
            task_grads = torch.autograd.grad(loss_meta,
                                             model_outer.weights + model_outer.biases + [model_outer.task_context])
            for i in range(len(model_inner.weights + model_inner.biases) + 1):
                meta_gradient[i] += task_grads[i].detach()
        
        # mean_tasks_mse = norm_batch_mse.mean()
        # norm_batch_mse = norm_batch_mse-mean_tasks_mse
        
        # ------------ meta update ------------
        meta_optimiser.zero_grad()

        # assign meta-gradient
        for i in range(len(model_outer.weights)):
            model_outer.weights[i].grad = meta_gradient[i] / args.tasks_per_metaupdate
            meta_gradient[i] = 0
        for j in range(len(model_outer.biases)):
            model_outer.biases[j].grad = meta_gradient[i + j + 1] / args.tasks_per_metaupdate
            meta_gradient[i + j + 1] = 0
        model_outer.task_context.grad = meta_gradient[i + j + 2] / args.tasks_per_metaupdate
        meta_gradient[i + j + 2] = 0

        # do update step on outer model
        meta_optimiser.step()
        
        
        ###################################################################################
            # update the Distribution Adversary --> Apply REINFORCE optimization
        ###################################################################################
        
        if game_framework:
            if i_iter % dist_interval == 0:
                mean_tasks_mse = norm_batch_mse.mean()
                norm_batch_mse = norm_batch_mse - mean_tasks_mse
                
                log_det_forward_total = log_det - log_det_forward - log_det_norm
                dist_loss_first = torch.mean(norm_batch_mse[:, None].detach()*log_det_forward_total[:, None])        
                dist_loss_second = log_p_phi

                dist_optimiser.zero_grad()
                dist_loss = dist_loss_first - args.entropy_weight * dist_loss_second
                dist_losses = -dist_loss

                # print('iter', i_iter)
                # print('dist_loss_first', dist_loss_first)
                # print('dist_loss_second', dist_loss_second)
                # print('dist_losses', dist_losses)

                dist_losses_mean.append(-dist_losses.detach().item())
                dist_losses_max.append(torch.max(dist_loss).detach().item())
                dist_losses_std.append(torch.std(dist_loss).detach().item())
                dist_losses_kl.append(dist_loss_second.detach().item())

                dist_losses.backward()
                torch.nn.utils.clip_grad_norm_(dist_model.parameters(), max_norm=5)
                dist_optimiser.step()

                lrs.append(dist_optimiser.param_groups[0]['lr'])
                dist_scheduler.step()

                # calculate the entropy
                if args.init_dis == 'Uniform':
                    entropy_first = torch.log((high_bound[0]-low_bound[0]) * (high_bound[1]-low_bound[1]))
                if args.init_dis == 'Normal':
                    entropy_first = torch.log(2 * np.pi * np.e * sigma[0] * sigma[1])
                log_det_sum = log_det_forward + log_det_norm
                entropy_second = log_det_sum.mean()
                entropy = entropy_first + entropy_second
                dist_entropy.append(entropy.detach().item())
                dist_entropy_forward.append(log_det_forward.mean().detach().item())
                dist_entropy_norm.append(log_det_norm.mean().detach().item())
        
        train_losses_mean.append(np.mean(loss_batch_mse))
        train_losses_max.append(np.max(loss_batch_mse))
        train_losses_std.append(np.std(loss_batch_mse))
        train_losses_cvar.append(cal_cvar(loss_batch_mse))

        if i_iter % log_interval == 0:
            if game_framework:
                loss_init_val_mean, loss_init_val_max, loss_init_val_cvar = eval_init(args, copy.deepcopy(model_outer), task_family_test, game_framework, n_tasks=500)
                val_init_losses_mean.append(loss_init_val_mean)
                val_init_losses_max.append(loss_init_val_max)
                val_init_losses_cvar.append(loss_init_val_cvar)
            
            loss_val, loss_val_max, loss_val_std, loss_val_cvar = eval_iter(args, copy.deepcopy(dist_model), copy.deepcopy(model_outer), task_family_test, game_framework, n_tasks=500)
            val_losses_mean.append(loss_val)
            val_losses_max.append(loss_val_max)
            val_losses_std.append(loss_val_std)
            val_losses_cvar.append(loss_val_cvar)

            if loss_val < min_loss:
                min_loss = loss_val
                print('save model')
                torch.save(copy.deepcopy(model_outer), args.task + '_logs_model/' + exp_string_outer)
                if game_framework:
                    torch.save(copy.deepcopy(dist_model), args.task + '_logs_model/' + exp_string_dist)
            # print('iter', i_iter, 'train_loss', np.mean(loss_batch_mse), 'dist_loss', -dist_losses.detach().item(), 'eval_loss', loss_val, 'eval_init_loss', loss_init_val_mean)
            print('iter', i_iter, 'train_loss', np.mean(loss_batch_mse), 'eval_loss', loss_val)
        
            if args.run_type == 'armaml':
                os.makedirs(args.task + "_result_files/armaml_result_files", exist_ok=True)
                np.savetxt(args.task + "_result_files/armaml_result_files/train_mean_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) + ".csv", train_losses_mean, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/train_max_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", train_losses_max, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/train_std_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", train_losses_std, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/train_cvar_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", train_losses_cvar, delimiter=",")

                np.savetxt(args.task + "_result_files/armaml_result_files/val_mean_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight)+ ".csv", val_losses_mean, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/val_max_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", val_losses_max, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/val_std_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", val_losses_std, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/val_cvar_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", val_losses_cvar, delimiter=",")

                np.savetxt(args.task + "_result_files/armaml_result_files/val_init_mean_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) + ".csv", val_init_losses_mean, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/val_init_max_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", val_init_losses_max, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/val_init_cvar_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", val_init_losses_cvar, delimiter=",")

                np.savetxt(args.task + "_result_files/armaml_result_files/dist_mean_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) + ".csv", dist_losses_mean, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_max_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", dist_losses_max, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_std_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", dist_losses_std, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_entropy_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", dist_entropy, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_lr_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", lrs, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_entropy_forward_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", dist_entropy_forward, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_entropy_norm_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", dist_entropy_norm, delimiter=",")
                np.savetxt(args.task + "_result_files/armaml_result_files/dist_loss_kl_armaml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + str(args.num_latent_layers) + '_lr' + str(args.lr_dist) + '_entropy' + str(args.entropy_weight) +".csv", dist_losses_kl, delimiter=",")
                
                print('finish armaml')
            
            if args.run_type == 'maml':
                os.makedirs(args.task + "_result_files/maml_result_files", exist_ok=True)
                np.savetxt(args.task + "_result_files/maml_result_files/train_mean_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) + ".csv", train_losses_mean, delimiter=",")
                np.savetxt(args.task + "_result_files/maml_result_files/train_max_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) +".csv", train_losses_max, delimiter=",")
                np.savetxt(args.task + "_result_files/maml_result_files/train_std_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) +".csv", train_losses_std, delimiter=",")
                np.savetxt(args.task + "_result_files/maml_result_files/train_cvar_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) +".csv", train_losses_cvar, delimiter=",")

                np.savetxt(args.task + "_result_files/maml_result_files/val_mean_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) + ".csv", val_losses_mean, delimiter=",")
                np.savetxt(args.task + "_result_files/maml_result_files/val_max_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) +".csv", val_losses_max, delimiter=",")
                np.savetxt(args.task + "_result_files/maml_result_files/val_std_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) +".csv", val_losses_std, delimiter=",")
                np.savetxt(args.task + "_result_files/maml_result_files/val_cvar_maml_"+ args.init_dis + str(args.dist_interval) +str(args.n_iter)+ args.flow_type + '_lr' + str(args.lr_dist) +".csv", val_losses_cvar, delimiter=",")
                print('finish maml')

def eval_iter(args, dist_model, model, task_family, game_framework, n_tasks=500, return_gradnorm=False):    
    # copy weights of network
    if args.task == 'sine': 
        n_tasks = 500
    else:
        n_tasks = 100
    
    copy_weights = [w.clone() for w in model.weights]
    copy_biases = [b.clone() for b in model.biases]
    copy_context = model.task_context.clone()

    # get the task family (with infinite number of tasks)
    losses = []
    gradnorms = []
    
    init_param_tensor = task_family.sample_init_param(n_tasks, args.init_dis).cuda()
    
    if game_framework:
        # generate hyper params after transformation as norm_z
        norm_z = dist_model(init_param_tensor, train=False)
    else:
        norm_z = init_param_tensor

    if args.task == 'sine': 
        x_all, y_all = task_family.generate_tasks(n_tasks, args.update_batch_size+1000, norm_z)
    else:
        x_all, y_all = task_family.generate_tasks(n_tasks, args.task_num_data_points, norm_z.cpu().detach().numpy())
    
    x_c, y_c, x_t, y_t = task_family.sample_meta_dataset(x_all, y_all, args.update_batch_size)
    x_c, y_c, x_t, y_t = x_c.cuda(), y_c.cuda(), x_t.cuda(), y_t.cuda() 

    for t in range(n_tasks):

        # reset network weights
        model.weights = [w.clone() for w in copy_weights]
        model.biases = [b.clone() for b in copy_biases]
        model.task_context = copy_context.clone()      
        
        # get data for current task
        curr_inputs, curr_targets = x_c[t,:,:], y_c[t,:,:]

        # ------------ update on current task ------------

        for _ in range(1, args.num_inner_updates + 1):

            curr_outputs = model(curr_inputs, args.task)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # update task parameters
            params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
            grads = torch.autograd.grad(task_loss, params)

            gradnorms.append(np.mean(np.array([g.norm().item() for g in grads])))

            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] - args.lr_inner * grads[i].detach()
            for j in range(len(model.biases)):
                model.biases[j] = model.biases[j] - args.lr_inner * grads[i + j + 1].detach()
            model.task_context = model.task_context - args.lr_inner * grads[i + j + 2].detach()

        # ------------ logging ------------

        # compute true loss on entire input range
        losses.append(F.mse_loss(model(x_t[t,:,:], args.task), y_t[t,:,:]).detach().item())

    # reset network weights
    model.weights = [w.clone() for w in copy_weights]
    model.biases = [b.clone() for b in copy_biases]
    model.task_context = copy_context.clone()

    losses_mean = np.mean(losses)
    losses_max = np.max(losses)
    losses_std = np.std(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    losses_cvar= cal_cvar(losses)
    
    if not return_gradnorm:
        return losses_mean, losses_max, losses_std, losses_cvar
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)


# log in the val_loss in initial distribution
def eval_init(args, model, task_family, game_framework, n_tasks=500, return_gradnorm=False):
    if args.task == 'sine': 
        n_tasks = 500
    else:
        n_tasks = 100
    
    copy_weights = [w.clone() for w in model.weights]
    copy_biases = [b.clone() for b in model.biases]
    copy_context = model.task_context.clone()

    # get the task family (with infinite number of tasks)
    losses = []
    gradnorms = []
    
    init_param_tensor = task_family.sample_init_param(n_tasks, args.init_dis)
    
    if game_framework:
        if args.task == 'sine': 
            x_all, y_all = task_family.generate_tasks(n_tasks, args.update_batch_size+1000, init_param_tensor)
        else:
            x_all, y_all = task_family.generate_tasks(n_tasks, args.task_num_data_points, init_param_tensor.cpu().detach().numpy())
        
        x_c, y_c, x_t, y_t = task_family.sample_meta_dataset(x_all, y_all, args.update_batch_size)
        x_c, y_c, x_t, y_t = x_c.cuda(), y_c.cuda(), x_t.cuda(), y_t.cuda() 

        for t in range(n_tasks):

            # reset network weights
            model.weights = [w.clone() for w in copy_weights]
            model.biases = [b.clone() for b in copy_biases]
            model.task_context = copy_context.clone()      
            
            # get data for current task
            curr_inputs, curr_targets = x_c[t,:,:], y_c[t,:,:]

            # ------------ update on current task ------------

            for _ in range(1, args.num_inner_updates + 1):

                curr_outputs = model(curr_inputs, args.task)

                # compute loss for current task
                task_loss = F.mse_loss(curr_outputs, curr_targets)

                # update task parameters
                params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
                grads = torch.autograd.grad(task_loss, params)

                gradnorms.append(np.mean(np.array([g.norm().item() for g in grads])))

                for i in range(len(model.weights)):
                    model.weights[i] = model.weights[i] - args.lr_inner * grads[i].detach()
                for j in range(len(model.biases)):
                    model.biases[j] = model.biases[j] - args.lr_inner * grads[i + j + 1].detach()
                model.task_context = model.task_context - args.lr_inner * grads[i + j + 2].detach()

            # ------------ logging ------------

            # compute true loss on entire input range
            losses.append(F.mse_loss(model(x_t[t,:,:], args.task), y_t[t,:,:]).detach().item())

        losses_mean = np.mean(losses)
        losses_max = np.max(losses)
        losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
        losses_cvar= cal_cvar(losses)
        
        if not return_gradnorm:
            return losses_mean, losses_max, losses_cvar
        else:
            return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)


def eval(args, exp_string_dist, exp_string_outer, n_tasks=500, return_gradnorm=False):
    print('transformed_dis:', args.transformed_dis)
    print('exp_string_dist', exp_string_dist)
    print('exp_string_outer', exp_string_outer)
    if args.task == 'sine':
        task_family_test = tasks_sine.RegressionTasksSinusoidal()
        n_tasks = 500
        low_bound = torch.tensor([0.1, 0.0]) # Lower bounds for each dimension
        high_bound = torch.tensor([5.0, np.pi])  # Upper bounds for each dimension
        mu = torch.tensor([2.5, 1.5])
        sigma = torch.tensor([0.8, 0.5])
    elif args.task == 'acrobot':
        task_family_test = tasks_acrobot.AcrobotDataset()
        n_tasks = 100
        low_bound = torch.tensor([0.4, 0.4]) # Lower bounds for each dimension
        high_bound = torch.tensor([1.6, 1.6])  # Upper bounds for each dimension
        mu = torch.tensor([1.0, 1.0])
        sigma = torch.tensor([0.2, 0.2])
    elif args.task == 'pendulum':
        task_family_test = tasks_pendulum.PendulumDataset()
        n_tasks = 100
        low_bound = torch.tensor([0.4, 0.4]) # Lower bounds for each dimension
        high_bound = torch.tensor([1.6, 1.6])  # Upper bounds for each dimension
        mu = torch.tensor([1.0, 1.0])
        sigma = torch.tensor([0.2, 0.2])
    
    model = torch.load(args.task + '_logs_model/' + exp_string_outer)
    model.eval()

    # copy weights of network
    copy_weights = [w.clone() for w in model.weights]
    copy_biases = [b.clone() for b in model.biases]
    copy_context = model.task_context.clone()

    # get the task family (with infinite number of tasks)
    losses = []
    gradnorms = []

    if args.transformed_dis:
        dist_model = torch.load(args.task + '_logs_model/' + exp_string_dist)
        dist_model.eval()
        # --- inner loop ---

        init_param_tensor = task_family_test.sample_init_param(n_tasks, args.init_dis)
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_armaml_init_param_m_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", init_param_tensor[:, 0], delimiter=",")
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_armaml_init_param_l_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", init_param_tensor[:, 1], delimiter=",")

        # if args.init_dis == 'Uniform':
        #     q0 = Uniform(shape=1, low=low_bound, high=high_bound)
        #     log_det = q0.log_prob(init_param_tensor.cuda())
        # elif args.init_dis == 'Normal':
        #     q0 = GaussianDistribution(loc=mu, scale=sigma)
        #     log_det = q0.log_prob(init_param_tensor.cuda())
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_armaml_init_param_probability"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", log_det.exp().cpu().detach().numpy(), delimiter=",")

        # generate hyper params after transformation as norm_z
        z, norm_z, log_det, log_det_forward, log_det_norm, log_p_phi = dist_model.forward(init_param_tensor, train=True)
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_armaml_dist_param_m_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", norm_z[:, 0].cpu().detach().numpy(), delimiter=",")
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_armaml_dist_param_l_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", norm_z[:, 1].cpu().detach().numpy(), delimiter=",")
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_armaml_dist_probability"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", log_det.exp().cpu().detach().numpy(), delimiter=",")
    
        if args.task == 'sine': 
            x_all, y_all = task_family_test.generate_tasks(n_tasks, args.update_batch_size+1000, norm_z)
        else:
            x_all, y_all = task_family_test.generate_tasks(n_tasks, args.task_num_data_points, norm_z.cpu().detach().numpy())
    
    else:     
        init_param_tensor = task_family_test.sample_init_param(n_tasks, args.init_dis)
        # np.savetxt(args.task + "_result_files/test_initdis_armaml_init_param_amp_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + ".csv", init_param_tensor[:, 0], delimiter=",")
        # np.savetxt(args.task + "_result_files/test_initdis_armaml_init_param_pha_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + ".csv", init_param_tensor[:, 1], delimiter=",")
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_initdis_armaml_init_param_amp_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", init_param_tensor[:, 0], delimiter=",")
        # np.savetxt(args.task + "_result_files/armaml_result_files/test_initdis_armaml_init_param_pha_"+ args.init_dis + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", init_param_tensor[:, 1], delimiter=",")

        if args.task == 'sine': 
            x_all, y_all = task_family_test.generate_tasks(n_tasks, args.update_batch_size+1000, init_param_tensor)
        else:
            x_all, y_all = task_family_test.generate_tasks(n_tasks, args.task_num_data_points, init_param_tensor)
    

    x_c, y_c, x_t, y_t = task_family_test.sample_meta_dataset(x_all, y_all, args.update_batch_size)
    x_c, y_c, x_t, y_t = x_c.cuda(), y_c.cuda(), x_t.cuda(), y_t.cuda() 
            
    for t in range(n_tasks):

        # reset network weights
        model.weights = [w.clone() for w in copy_weights]
        model.biases = [b.clone() for b in copy_biases]
        model.task_context = copy_context.clone()      
        
        # get data for current task
        curr_inputs, curr_targets = x_c[t,:,:], y_c[t,:,:]

        # ------------ update on current task ------------

        for _ in range(1, args.num_inner_updates + 1):

            curr_outputs = model(curr_inputs, args.task)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # update task parameters
            params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
            grads = torch.autograd.grad(task_loss, params)

            gradnorms.append(np.mean(np.array([g.norm().item() for g in grads])))

            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] - args.lr_inner * grads[i].detach()
            for j in range(len(model.biases)):
                model.biases[j] = model.biases[j] - args.lr_inner * grads[i + j + 1].detach()
            model.task_context = model.task_context - args.lr_inner * grads[i + j + 2].detach()

        # ------------ logging ------------

        # compute true loss on entire input range
        losses.append(F.mse_loss(model(x_t[t,:,:], args.task), y_t[t,:,:]).detach().item())

    # if args.transformed_dis:
    #     np.savetxt(args.task + "_result_files/armaml_result_files/test_losses_armaml"+ args.init_dis +str(args.log_number)+ '_n' + str(args.n_iter)+ args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", losses, delimiter=",")
    # else:
    #     np.savetxt(args.task + "_result_files/armaml_result_files/test_initdis_losses_armaml"+ args.init_dis +str(args.log_number)+ '_n' + str(args.n_iter)+ args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + ".csv", losses, delimiter=",")

    losses_mean = np.mean(losses)
    # losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    losses_cvar = cal_cvar(losses)
    print(losses_mean)
    print(losses_cvar)
    
