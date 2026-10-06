#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:19:54 2026

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
                    default='random_walk',
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
ablation_sizes = [0, 10, 20, 40]
n_sizes = len(ablation_sizes)
shift_mode = 'temporal'
band_thresh_percentile = 95

rel_grid_score_band_ablation_decoding_error = np.zeros((n_seeds, n_sizes))
band_score_band_decoding_error = np.zeros((n_seeds, n_sizes))

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

for ss in range(n_seeds):
        print(ss)
        # Loading band cell data
        band_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + 'analysis_outputs/border_band/'     
        X = np.load(band_cell_analysis_path + 'band_scores.npz')
        band_scores = X['band_scores']
        band_scores[np.isnan(band_scores)] = 0
        band_thresh = np.percentile(band_scores, band_thresh_percentile)
        band_score_sorted_ids = np.argsort(band_scores)
        band_ids = np.argwhere(band_scores > band_thresh).flatten()
        band_score_sorted_ids = band_ids[np.argsort(band_scores[band_ids])]
        
        # Loading grid cell data
        grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/' + shift_mode + ' shift/predictive_retrospective/'          
        X = np.load(grid_cell_analysis_path + 'final_model.pth_' + options.trajectory_style + '_summary_data.npz')
        dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + options.trajectory_style + '.npy')
        grid_scores = X['zero_scores'].T
        max_shifted_grid_scores = X['best_scores']
        best_shift_ids = X['best_idx']
        rel_grid_score_improvement = (max_shifted_grid_scores - grid_scores) / np.abs(grid_scores)
        rel_grid_score_improvement[best_shift_ids <= 0] = 0
        rel_grid_score_improvement_sorted_ids = band_ids[np.argsort(rel_grid_score_improvement[band_ids])]
         
        # Loading model
        model_name = 'final_model.pth'
        model_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name
        
        # Ablating band units by band score
        for nn in range(n_sizes):
            model = RNN(options, place_cells)
            model = model.to(options.device)
            saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
        
            if nn == 0:
                ablation_ids = []
            else:
                ablation_ids = band_score_sorted_ids[-int(ablation_sizes[nn]):]

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
            band_score_band_decoding_error[ss, nn] = np.mean(e)
            
        # Ablating band units by relative grid score improvement
        for nn in range(n_sizes):
            model = RNN(options, place_cells)
            model = model.to(options.device)
            saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
        
            if nn == 0:
                ablation_ids = []
            else:
                ablation_ids = rel_grid_score_improvement_sorted_ids[-int(ablation_sizes[nn]):]

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
            rel_grid_score_band_ablation_decoding_error[ss, nn] = np.mean(e)