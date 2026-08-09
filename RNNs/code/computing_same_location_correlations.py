#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:00:36 2026

@author: wredman
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
from utils import generate_run_ID
import random
from tqdm import tqdm
import scipy

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
                    default=20000, # 20000
                    help='number of trajectories per batch') 
parser.add_argument('--sequence_length',
                    default=20,
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
                    default= None,
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
res = 44
n_pairs = 20
timestep = 15
border_buffer = 0.2 # 0.2
border_buffer_bins = int(np.ceil(border_buffer / (options.box_width / res)))
shift_mode = 'spatial'

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'
grid_correlations = np.zeros((n_seeds, res, res, n_pairs)) * np.nan
predictive_grid_correlations = np.zeros((n_seeds, res, res, n_pairs)) * np.nan
movement_dist = np.zeros((n_seeds, res, res, n_pairs)) * np.nan

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
        
    model_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name
    model_name = 'final_model.pth'
    model = RNN(options, place_cells)
    model = model.to(options.device)
    saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model)    
    inputs, p, pc_outputs = trajectory_generator.get_test_batch()
    v = inputs[0].detach().numpy()
    pred_p = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()

    e = torch.sqrt(((p - pred_p)**2).sum(-1)).median(dim = 0).values
    e = e.numpy()
    print(np.mean(e))
    
    g = model.g(inputs)
    g = g.detach().numpy()
        
    p = p[timestep, :, :]
    g = g[timestep, :, :]
    v = v[timestep, :, :]
    p += (options.box_width/2)
    binned_pos = np.floor((p / options.box_width) * res)
    # binned_pos = binned_pos.reshape(-1, np.shape(binned_pos)[2])
    # g = g.reshape(-1, np.shape(g)[2])

    for ii in range(border_buffer_bins, res - border_buffer_bins + 1):
        for jj in range(border_buffer_bins, res - border_buffer_bins + 1):
            ids = np.argwhere((binned_pos[:, 0] == ii) & (binned_pos[:, 1] == jj)).flatten()
            
            if n_pairs < scipy.special.comb(len(ids), 2):
                for kk in range(n_pairs):
                    pair = np.random.randint(len(ids), size = (2))
                    grid_correlations[ss, ii, jj, kk] = scipy.stats.pearsonr(g[ids[pair[0]], grid_ids], g[ids[pair[1]], grid_ids]).statistic
                    predictive_grid_correlations[ss, ii, jj, kk] = scipy.stats.pearsonr(g[ids[pair[0]], predictive_ids], g[ids[pair[1]], predictive_ids]).statistic
                    movement_dist[ss, ii, jj, kk] = np.sqrt(np.sum((v[ids[pair[0]], :] - v[ids[pair[1]], :])**2))
                    
# Plotting 
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/grid_predictive_retrospective/' + shift_mode + ' shift/'
fig_save = False

X_grid = grid_correlations.flatten()
X_grid = X_grid[~np.isnan(X_grid)]
X_pred_grid = predictive_grid_correlations.flatten()
X_pred_grid = X_pred_grid[~np.isnan(X_pred_grid)]

movement_dist = movement_dist.flatten()
movement_dist = movement_dist[~np.isnan(movement_dist)]
predictive_grid_corr = predictive_grid_correlations.flatten()
predictive_grid_corr = predictive_grid_corr[~np.isnan(predictive_grid_corr)]
grid_corr = grid_correlations.flatten()
grid_corr = grid_corr[~np.isnan(grid_corr)]

figure = plt.figure(figsize = (6, 4))
plt.hist(X_grid, label='Grid', alpha = 0.3, bins = np.arange(0.6, 1.025, 0.025), density = True)
plt.hist(X_pred_grid, label='Predictive grid', alpha = 0.3, bins = np.arange(0.6, 1.025, 0.025), density = True)         
plt.xlabel('Population correlation')
plt.legend()
if fig_save: 
    plt.savefig(save_path + 'grid_predictive_grid_population_correlation_distribution.png')
    plt.savefig(save_path + 'grid_predictive_grid_population_correlation_distribution.svg', format = 'svg')

print(np.nanmedian(X_grid))
print(np.nanmedian(X_pred_grid))

figure = plt.figure(figsize = (8, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.nanmedian(grid_correlations, axis = (0, 3)))
plt.title('Grid units')
plt.clim([0.5, 1.0])
plt.subplot(1, 2, 2)
plt.imshow(np.nanmedian(predictive_grid_correlations, axis = (0, 3)))
plt.title('Predictive grid units')
plt.clim([0.5, 1.0])

figure = plt.figure(figsize = (8, 4))
plt.subplot(1, 2, 1)
plt.plot(movement_dist.flatten()[np.arange(0, len(X_grid), 100)], X_grid[np.arange(0, len(X_grid), 100)], 'ko')
plt.xlabel('Euclidean distance')
plt.ylabel('Correlation')
plt.title('Movement dist vs. grid unit corr')
plt.axis([0, 0.1, 0, 1.0])
plt.subplot(1, 2, 2)
plt.plot(movement_dist.flatten()[np.arange(0, len(X_grid), 100)], X_pred_grid[np.arange(0, len(X_grid), 100)], 'ko')
plt.title('Movement dist vs. predictive grid unit corr')
plt.xlabel('Euclidean distance')
plt.ylabel('Correlation')
plt.axis([0, 0.1, 0, 1.0])
if fig_save: 
    plt.savefig(save_path + 'grid_predictive_grid_population_correlation_vs_movement_vector_dist.png')
    plt.savefig(save_path + 'grid_predictive_grid_population_correlation_vs_movement_vector_dist.svg', format = 'svg')

movement_dist_bins = 5
binned_movement_dist = np.percentile(movement_dist, [np.arange(0, 100 + movement_dist_bins, movement_dist_bins)])[0]
median_binned_grid_correlation = np.zeros(len(binned_movement_dist) - 1)
percentile_binned_grid_correlation = np.zeros((2, len(binned_movement_dist) - 1))
median_binned_predictive_grid_correlation = np.zeros(len(binned_movement_dist) - 1)
percentile_binned_predictive_grid_correlation = np.zeros((2, len(binned_movement_dist) - 1))

for ii in range(len(binned_movement_dist) - 1):
    bin_ids = np.argwhere((movement_dist > binned_movement_dist[ii]) & (movement_dist < binned_movement_dist[ii + 1]))
    median_binned_grid_correlation[ii] = np.nanmedian(grid_corr[bin_ids])   
    median_binned_predictive_grid_correlation[ii] = np.nanmedian(predictive_grid_corr[bin_ids])
    if len(bin_ids) > 0:
        percentile_binned_predictive_grid_correlation[0, ii] = np.percentile(predictive_grid_corr[bin_ids], 25)
        percentile_binned_predictive_grid_correlation[1, ii] = np.percentile(predictive_grid_corr[bin_ids], 75)
        percentile_binned_grid_correlation[0, ii] = np.percentile(grid_corr[bin_ids], 25)
        percentile_binned_grid_correlation[1, ii] = np.percentile(grid_corr[bin_ids], 75)

x = (binned_movement_dist[1:] + binned_movement_dist[:-1]) / 2
figure = plt.figure(figsize = (6, 4))
for ii in range(len(binned_movement_dist) - 1):
    plt.plot([x[ii], x[ii]], percentile_binned_grid_correlation[:, ii], 'k-')
    plt.plot([x[ii], x[ii]], percentile_binned_predictive_grid_correlation[:, ii], 'r-')
plt.plot(x, median_binned_grid_correlation, 'ko', label ='Grid')
plt.plot(x, median_binned_predictive_grid_correlation, 'ro', label ='Predictive grid')
plt.axis([0.0, 0.09, 0.7, 1.0])
if fig_save: 
    plt.savefig(save_path + 'grid_predictive_grid_population_correlation_vs_movement_vector_dist_binned.png')
    plt.savefig(save_path + 'grid_predictive_grid_population_correlation_vs_movement_vector_dist_binned.svg', format = 'svg')



