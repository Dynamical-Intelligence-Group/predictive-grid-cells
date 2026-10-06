#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:23:04 2026

@author: wredman
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
from utils import generate_run_ID
import random
from tqdm import tqdm

# Parameters necessary to evaluate network
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=1,
                    help='number of training epochs') #100
parser.add_argument('--n_steps',
                    default= 1,
                    help='batches per epoch') #1000
parser.add_argument('--batch_size',
                    default=1000,
                    help='number of trajectories per batch') #200
parser.add_argument('--sequence_length',
                    default=40,
                    help='number of steps in trajectory') #20
parser.add_argument('--learning_rate',
                    default=1e-4,
                    help='gradient descent learning rate') #1e-4
parser.add_argument('--Np',
                    default=512, 
                    help='number of place cells') #512
parser.add_argument('--Ng',
                    default=4096,
                    help='number of grid cells') #4096
parser.add_argument('--place_cell_rf',
                    default=0.12,
                    help='width of place cell center tuning curve (m)') #0.12
parser.add_argument('--surround_scale',
                    default=2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    default=1e-4,
                    help='strength of weight decay on recurrent weights') #1e-4
parser.add_argument('--DoG',
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default= 2.2, 
                    help='width of training environment') #2.2
parser.add_argument('--box_height',
                    default=2.2, 
                    help='height of training environment') #2.2
parser.add_argument('--device',
                    default='cpu',
                    help='device to use for training')
parser.add_argument('--trajectory_style',
                    default='straight',
                    choices=['random_walk', 'straight', 'per_step_random'],
                    help='motion regime: smooth random walk (default), straight with fixed speed, or new random heading/speed each step')
parser.add_argument('--trajectory_fixed_speed',
                    default=None,
                    type=float,
                    help='fixed forward speed in m/s when using --trajectory_style straight')
parser.add_argument('--trajectory_dt',
                    default=0.02,
                    type=float,
                    help='trajectory timestep (seconds)')
parser.add_argument('--trajectory_turn_sigma_scale',
                    default=1.0,
                    type=float,
                    help='scale on rotational noise; reduce for more predictable heading')
parser.add_argument('--trajectory_speed_scale',
                    default=1.0,
                    type=float,
                    help='scale on forward speed; increase for faster motion')
parser.add_argument('--trajectory_speed_max',
                    default=None,
                    type=float,
                    help='optional cap on forward speed (m/s)')
parser.add_argument('--trajectory_velocity_smoothing',
                    default=0.0,
                    type=float,
                    help='EMA factor in [0,1); >0 smooths speed changes')
parser.add_argument('--trajectory_border_region',
                    default=0.03,
                    type=float,
                    help='distance from wall (m) that triggers avoidance turn/slowdown')
parser.add_argument('--trajectory_wall_slowdown',
                    default=0.25,
                    type=float,
                    help='speed multiplier near walls (non-periodic envs)')
parser.add_argument('--trajectory_wall_turn_scale',
                    default=1.0,
                    type=float,
                    help='strength of wall-induced turn when non-periodic')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

random.seed(0)

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

# Loading saved data
n_seeds = 10
n_units = options.Ng
ablation_sizes = [30, 60, 90, 120, 150]
n_sizes = len(ablation_sizes)
n_ablations = 20
shift_mode = 'temporal'

predictive_ablation_decoding_error = np.zeros((n_seeds, n_sizes, n_ablations))
predictive_ablation_decoding_error_across_trajectory = np.zeros((n_seeds, n_sizes, n_ablations, options.sequence_length))
retrospective_ablation_decoding_error = np.zeros((n_seeds, n_sizes, n_ablations))
retrospective_ablation_decoding_error_across_trajectory = np.zeros((n_seeds, n_sizes, n_ablations, options.sequence_length))
grid_ablation_decoding_error = np.zeros((n_seeds, n_sizes, n_ablations))
grid_ablation_decoding_error_across_trajectory = np.zeros((n_seeds, n_sizes, n_ablations, options.sequence_length))
random_ablation_decoding_error = np.zeros((n_seeds, n_sizes, n_ablations))
random_ablation_decoding_error_across_trajectory = np.zeros((n_seeds, n_sizes, n_ablations, options.sequence_length))

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

for ss in range(n_seeds):
        print(ss)
        # Loading normal, predictive, and retrospective grid cell data
        grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/' + shift_mode + ' shift/predictive_retrospective/'          
        X = np.load(grid_cell_analysis_path + 'final_model.pth_' + options.trajectory_style + '_summary_data.npz')

        grid_ids = np.load(grid_cell_analysis_path + 'grid_ids_' + options.trajectory_style + '.npy')
        predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + options.trajectory_style + '.npy')
        retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + options.trajectory_style + '.npy')
        dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + options.trajectory_style + '.npy')
        non_dead_ids = np.arange(0, 4096, 1)[dead_unit_ids == False]
        
        model_name = 'final_model.pth'
        model_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name
        
        # Ablating predictive grid cells
        for nn in range(n_sizes):
            for aa in range(n_ablations):
                model = RNN(options, place_cells)
                model = model.to(options.device)
                saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
        
                np.random.shuffle(predictive_ids)
                ablation_ids = predictive_ids[:int(ablation_sizes[nn])]

                saved_model['encoder.weight'][ablation_ids, :] = 0  
                saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
                saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
                saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
                saved_model['decoder.weight'][:, ablation_ids] = 0
                
                model.load_state_dict(saved_model)

                inputs, p, pc_outputs = trajectory_generator.get_test_batch()
                pred_p = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
                
                e_trajectory = torch.sqrt(((p - pred_p)**2).sum(-1))
                e = e_trajectory.median(dim = 0).values
                e = e.numpy()
                predictive_ablation_decoding_error[ss, nn, aa] = np.mean(e)
                predictive_ablation_decoding_error_across_trajectory[ss, nn, aa, :] = np.mean(e_trajectory.numpy(), axis = 1)
                
            # Ablating retrospective grid cells
            for nn in range(n_sizes):
                for aa in range(n_ablations):
                    model = RNN(options, place_cells)
                    model = model.to(options.device)
                    saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
            
                    np.random.shuffle(retrospective_ids)
                    if ablation_sizes[nn] == 'all':
                        ablation_ids = retrospective_ids
                    else:
                        ablation_ids = retrospective_ids[:int(ablation_sizes[nn])]
                    
                    saved_model['encoder.weight'][ablation_ids, :] = 0  
                    saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
                    saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
                    saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
                    saved_model['decoder.weight'][:, ablation_ids] = 0
                    
                    model.load_state_dict(saved_model)

                    inputs, p, pc_outputs = trajectory_generator.get_test_batch()
                    pred_p = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()

                    e_trajectory = torch.sqrt(((p - pred_p)**2).sum(-1))
                    e = e_trajectory.median(dim = 0).values
                    e = e.numpy()
                    retrospective_ablation_decoding_error[ss, nn, aa] = np.mean(e)
                    retrospective_ablation_decoding_error_across_trajectory[ss, nn, aa, :] = np.mean(e_trajectory.numpy(), axis = 1)
                    
        # Ablating grid cells
        for nn in range(n_sizes):
            for aa in range(n_ablations):
                 model = RNN(options, place_cells)
                 model = model.to(options.device)
                 saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
         
                 np.random.shuffle(grid_ids)
                 if ablation_sizes[nn] == 'all':
                     ablation_ids = grid_ids
                 else:
                     ablation_ids = grid_ids[:int(ablation_sizes[nn])]
                 
                 saved_model['encoder.weight'][ablation_ids, :] = 0  
                 saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
                 saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
                 saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
                 saved_model['decoder.weight'][:, ablation_ids] = 0
                 
                 model.load_state_dict(saved_model)

                 inputs, p, pc_outputs = trajectory_generator.get_test_batch()
                 pred_p = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()

                 e_trajectory = torch.sqrt(((p - pred_p)**2).sum(-1))
                 e = e_trajectory.median(dim = 0).values
                 e = e.numpy()
                 grid_ablation_decoding_error[ss, nn, aa] = np.mean(e)    
                 grid_ablation_decoding_error_across_trajectory[ss, nn, aa, :] = np.mean(e_trajectory.numpy(), axis = 1)

           
        # Ablating random units 
        for nn in range(n_sizes):
            for aa in range(n_ablations):
                 model = RNN(options, place_cells)
                 model = model.to(options.device)
                 saved_model = torch.load(model_path + model_name,  map_location=torch.device('cpu'))

                 np.random.shuffle(non_dead_ids)
                 ablation_ids = non_dead_ids[:int(ablation_sizes[nn])]

                 saved_model['encoder.weight'][ablation_ids, :] = 0  
                 saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
                 saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
                 saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
                 saved_model['decoder.weight'][:, ablation_ids] = 0
                   
                 model.load_state_dict(saved_model)

                 inputs, p, pc_outputs = trajectory_generator.get_test_batch()
                 pred_p = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()

                 e_trajectory = torch.sqrt(((p - pred_p)**2).sum(-1))
                 e = e_trajectory.median(dim = 0).values
                 e = e.numpy()
                 random_ablation_decoding_error[ss, nn, aa] = np.mean(e)
                 random_ablation_decoding_error_across_trajectory[ss, nn, aa, :] = np.mean(e_trajectory.numpy(), axis = 1)

                 
# Saving data
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/ablations_fixed_number/' + shift_mode + ' shift/'

np.save(save_path + 'predictive_ablations.npy', predictive_ablation_decoding_error)
np.save(save_path + 'retrospective_ablations.npy', retrospective_ablation_decoding_error)
np.save(save_path + 'grid_ablations.npy', grid_ablation_decoding_error)
np.save(save_path + 'random_ablations.npy', random_ablation_decoding_error)



