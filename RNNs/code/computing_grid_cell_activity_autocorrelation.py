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
import umap
import matplotlib
from sklearn.decomposition import PCA
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

def scatter3d(data, tags, ncols=2, nrows=2, s=1, alpha=0.5, azim_elev_title=True, edgecolor='none', **kwargs):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw={"projection": "3d"}, **kwargs)
    num_plots = ncols * nrows
    
    azims = np.linspace(0, 180, ncols + 1)[:-1]
    elevs = np.linspace(0, 90, nrows + 1)[:-1]
    view_angles = np.stack(np.meshgrid(azims, elevs), axis=-1).reshape(-1, 2)
    norm = matplotlib.colors.Normalize(np.amin(tags), np.amax(tags))
    color = matplotlib.cm.viridis(norm(tags))
    for i, ax in enumerate(axs.flat):
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], color = color, s=s, alpha=alpha, edgecolor =edgecolor)
        ax.azim = view_angles[i, 0]
        ax.elev = view_angles[i, 1]
        ax.axis("off")
        if azim_elev_title:
            ax.set_title(f"azim={ax.azim}, elev={ax.elev}")
    return fig, axs

# Defining cosine sim
def cos_sim(X, Y):
    nX = np.linalg.norm(X, axis = 2)
    nY = np.linalg.norm(Y, axis = 2)
    
    c = np.sum(X * Y, axis = 2) / (nX * nY)
    
    return c

# Loading saved data
n_seeds = 10
n_units = options.Ng
n_random_ablations = 1
max_shift = 30
shift_mode = 'temporal'
save_flag = False
#optimal_shifts_ablate_max = [8, 12, 20]
#optimal_shifts_ablate_min = [2, 9, 18]

#n_optimal_shifts = len(optimal_shifts_ablate_max)
if shift_mode == 'temporal':
    shifts = np.arange(-20, 21, 1)
else:
    shifts = np.arange(-0.4, 0.4, 0.02)

base_grid_cosine_sim = np.zeros((n_seeds, max_shift))
base_grid_cosine_sim_1_step = np.zeros((n_seeds, options.sequence_length - 1))
predictive_ablation_grid_cosine_sim = np.zeros((n_seeds, max_shift))
band_ablation_grid_cosine_sim = np.zeros((n_seeds, max_shift))
#predictive_optimal_shift_ablation_grid_cosine_sim = np.zeros((n_seeds, n_optimal_shifts, max_shift))
predictive_ablation_grid_cosine_sim_1_step = np.zeros((n_seeds, options.sequence_length - 1))
retrospective_ablation_grid_cosine_sim = np.zeros((n_seeds, max_shift))
retrospective_ablation_grid_cosine_sim_1_step = np.zeros((n_seeds, options.sequence_length - 1))
random_ablation_grid_cosine_sim = np.zeros((n_seeds, n_random_ablations, max_shift))
random_ablation_grid_cosine_sim_1_step = np.zeros((n_seeds, n_random_ablations,  options.sequence_length - 1))

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

band_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/border_band/' + shift_mode + ' shift/'     
band_units = np.load(band_cell_analysis_path + 'band_units.npy')

for ss in range(n_seeds):
        print(ss)
        # Loading normal, predictive, and retrospective grid units data
        grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/' + shift_mode + ' shift/predictive_retrospective/'          
        X = np.load(grid_cell_analysis_path + 'final_model.pth_' + options.trajectory_style + '_summary_data.npz')

        grid_ids = np.load(grid_cell_analysis_path + 'grid_ids_' + options.trajectory_style + '.npy')
        predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + options.trajectory_style + '.npy')
        retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + options.trajectory_style + '.npy')
        dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + options.trajectory_style + '.npy')
        non_dead_ids = np.arange(0, 4096, 1)[dead_unit_ids == False]
        non_grid_ids = np.setdiff1d(non_dead_ids, grid_ids)
        non_classified_ids = np.setdiff1d(non_grid_ids, predictive_ids)
        non_classified_ids = np.setdiff1d(non_classified_ids, retrospective_ids)
        
        optimal_shifts = shifts[X['best_idx']]
        
        # Loading band units data
        band_ids = np.argwhere(band_units[ss, :] == 1)
        non_classified_ids = np.setdiff1d(non_classified_ids, band_ids)
        
        # Setting up model
        model_name = 'final_model.pth'
        model_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name
        
        # Base model
        model = RNN(options, place_cells)
        model = model.to(options.device)
        saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
                
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        g = model.g(inputs)
        g = g.detach().numpy()  
                
        base_grid_cosine_sim[ss, 0] = 1 
        for tt in range(1, max_shift):
            if tt == 1:
                base_grid_cosine_sim_1_step[ss, :] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]), axis = (1, 2))
            base_grid_cosine_sim[ss, tt] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]))         
        
        # Ablating predictive grid cells
        model = RNN(options, place_cells)
        model = model.to(options.device)
        saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
        
        ablation_ids = predictive_ids 

        saved_model['encoder.weight'][ablation_ids, :] = 0  
        saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
        saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
        saved_model['decoder.weight'][:, ablation_ids] = 0
    
        model.load_state_dict(saved_model)

        #inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        g = model.g(inputs)
        g = g.detach().numpy()                     
        
        predictive_ablation_grid_cosine_sim[ss, 0] = 1 
        for tt in range(1, max_shift):
            if tt == 1:
                predictive_ablation_grid_cosine_sim_1_step[ss, :] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]), axis = (1, 2))
            predictive_ablation_grid_cosine_sim[ss, tt] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]))
                
        # Ablating band cells
        model = RNN(options, place_cells)
        model = model.to(options.device)
        saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
         
        ablation_ids = np.setdiff1d(band_ids, grid_ids)

        saved_model['encoder.weight'][ablation_ids, :] = 0  
        saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
        saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
        saved_model['decoder.weight'][:, ablation_ids] = 0
     
        model.load_state_dict(saved_model)

        #inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        g = model.g(inputs)
        g = g.detach().numpy()     
        
        band_ablation_grid_cosine_sim[ss, 0] = 1 
        for tt in range(1, max_shift):
            band_ablation_grid_cosine_sim[ss, tt] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]))
            
        # Ablating predictive grid cells based on their optimal shift      
        #for ii in range(n_optimal_shifts):
        #    model = RNN(options, place_cells)
        #    model = model.to(options.device)
        #    saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
            
        #    ablation_ids = predictive_ids[(optimal_shifts[predictive_ids] > optimal_shifts_ablate_min[ii]) & (optimal_shifts[predictive_ids] <= optimal_shifts_ablate_max[ii])]

        #    saved_model['encoder.weight'][ablation_ids, :] = 0  
        #    saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
        #    saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
        #    saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
        #    saved_model['decoder.weight'][:, ablation_ids] = 0
    
        #    model.load_state_dict(saved_model)

        #    inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        #    g = model.g(inputs)
        #    g = g.detach().numpy()     
                
        #    predictive_optimal_shift_ablation_grid_cosine_sim[ss, ii, 0] = 1 
        #    for tt in range(1, max_shift):
        #        predictive_optimal_shift_ablation_grid_cosine_sim[ss, ii, tt] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]))
                         
        # Ablating retrospective grid cells
        model = RNN(options, place_cells)
        model = model.to(options.device)
        saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
            
        ablation_ids = retrospective_ids
        saved_model['encoder.weight'][ablation_ids, :] = 0  
        saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
        saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
        saved_model['decoder.weight'][:, ablation_ids] = 0
                    
        model.load_state_dict(saved_model)

        #inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        g = model.g(inputs)
        g = g.detach().numpy()   
                                        
        retrospective_ablation_grid_cosine_sim[ss, 0] = 1 
        for tt in range(1, max_shift):
            if tt == 1:
                retrospective_ablation_grid_cosine_sim_1_step[ss, :] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]), axis = (1, 2))
            retrospective_ablation_grid_cosine_sim[ss, tt] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]))
                                 
        # Ablating random units 
        for aa in range(n_random_ablations):
            model = RNN(options, place_cells)
            model = model.to(options.device)
            saved_model = torch.load(model_path + model_name,  map_location=torch.device('cpu'))

            np.random.shuffle(non_grid_ids)
            ablation_ids = non_grid_ids[:int(len(predictive_ids))]

            saved_model['encoder.weight'][ablation_ids, :] = 0  
            saved_model['RNN.weight_ih_l0'][ablation_ids, :] = 0
            saved_model['RNN.weight_hh_l0'][:, ablation_ids] = 0
            saved_model['RNN.weight_hh_l0'][ablation_ids, :] = 0
            saved_model['decoder.weight'][:, ablation_ids] = 0
                   
            model.load_state_dict(saved_model)

            #inputs, p, pc_outputs = trajectory_generator.get_test_batch()
            g = model.g(inputs)
            g = g.detach().numpy()  
                 
            random_ablation_grid_cosine_sim[ss, aa, 0] = 1 
            for tt in range(1, max_shift):
                if tt == 1:
                    random_ablation_grid_cosine_sim_1_step[ss, aa, :] = np.median(cos_sim(g[:-tt, :, grid_ids], g[tt:, :, grid_ids]), axis = (1, 2))
                random_ablation_grid_cosine_sim[ss, aa, tt] = np.median(cos_sim(g[:-tt, :, np.setdiff1d(grid_ids, ablation_ids) ], g[tt:, :, np.setdiff1d(grid_ids, ablation_ids) ]))

# Saving data
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/ablations_fixed_number/' + shift_mode + ' shift/'
if save_flag:
    np.save(save_path + 'base_grid_cosine_sim_' + options.trajectory_style + '.npy', base_grid_cosine_sim)
    np.save(save_path + 'band_ablation_grid_cosine_sim_' + options.trajectory_style + '.npy', band_ablation_grid_cosine_sim)
    np.save(save_path + 'predictive_ablation_grid_cosine_sim_' + options.trajectory_style + '.npy', predictive_ablation_grid_cosine_sim)
    np.save(save_path + 'retrospective_ablation_grid_cosine_sim_' + options.trajectory_style + '.npy', retrospective_ablation_grid_cosine_sim)
    np.save(save_path + 'random_ablation_grid_cosine_sim_' + options.trajectory_style + '.npy', random_ablation_grid_cosine_sim)

# Plotting 
fig = plt.figure(figsize = (6, 4))
plt.fill_between(np.arange(0, max_shift, 1), np.percentile(base_grid_cosine_sim, 25, axis = 0), np.percentile(base_grid_cosine_sim, 75, axis = 0), color = 'k', alpha = 0.3)
plt.plot(np.median(base_grid_cosine_sim, axis = 0), 'k-', label = 'Base')
plt.fill_between(np.arange(0, max_shift, 1), np.percentile(predictive_ablation_grid_cosine_sim, 25, axis = 0), np.percentile(predictive_ablation_grid_cosine_sim, 75, axis = 0), color = 'g', alpha = 0.5)
plt.plot(np.median(predictive_ablation_grid_cosine_sim, axis = 0), 'g-', label = 'Predictive')
plt.fill_between(np.arange(0, max_shift, 1), np.percentile(retrospective_ablation_grid_cosine_sim, 25, axis = 0), np.percentile(retrospective_ablation_grid_cosine_sim, 75, axis = 0), color = 'r', alpha = 0.5)
plt.plot(np.median(retrospective_ablation_grid_cosine_sim, axis = 0), 'r-', label = 'Retrospective')
plt.fill_between(np.arange(0, max_shift, 1), np.percentile(band_ablation_grid_cosine_sim, 25, axis = 0), np.percentile(band_ablation_grid_cosine_sim, 75, axis = 0), color = 'y', alpha = 0.5)
plt.plot(np.median(band_ablation_grid_cosine_sim, axis = 0), 'y-', label = 'Band')
plt.fill_between(np.arange(0, max_shift, 1), np.percentile(np.median(random_ablation_grid_cosine_sim, axis = 1), 25, axis = 0), np.percentile(np.median(random_ablation_grid_cosine_sim, axis = 1), 75, axis = 0), color = 'b', alpha = 0.5)
plt.plot(np.median(np.median(random_ablation_grid_cosine_sim, axis = 1), axis = 0), 'b-', label = 'Random')
plt.legend()
plt.axis([0, 30, 0.4, 1.0])
if save_flag:
    plt.savefig(save_path + 'autocorrelation_vs_ablations_' + options.trajectory_style + '.png')
    plt.savefig(save_path + 'autocorrelation_vs_ablations_' + options.trajectory_style + '.svg', format = 'svg')

fig = plt.figure(figsize = (6, 4))
#plt.fill_between(np.arange(1, options.sequence_length, 1), np.percentile(base_grid_cosine_sim_1_step, 25, axis = 0), np.percentile(base_grid_cosine_sim_1_step, 75, axis = 0), color = 'k', alpha = 0.3)
plt.plot(np.median(base_grid_cosine_sim_1_step, axis = 0), 'ko-', label = 'Base')
#plt.fill_between(np.arange(1, options.sequence_length, 1), np.percentile(predictive_ablation_grid_cosine_sim_1_step, 25, axis = 0), np.percentile(predictive_ablation_grid_cosine_sim_1_step, 75, axis = 0), color = 'g', alpha = 0.5)
plt.plot(np.median(predictive_ablation_grid_cosine_sim_1_step, axis = 0), 'go-', label = 'Predictive')
#plt.fill_between(np.arange(1, options.sequence_length, 1), np.percentile(retrospective_ablation_grid_cosine_sim_1_step, 25, axis = 0), np.percentile(retrospective_ablation_grid_cosine_sim_1_step, 75, axis = 0), color = 'r', alpha = 0.5)
plt.plot(np.median(retrospective_ablation_grid_cosine_sim_1_step, axis = 0), 'ro-', label = 'Retrospective')
#plt.fill_between(np.arange(1, options.sequence_length, 1), np.percentile(np.median(random_ablation_grid_cosine_sim_1_step, axis = 1), 25, axis = 0), np.percentile(np.median(random_ablation_grid_cosine_sim_1_step, axis = 1), 75, axis = 0), color = 'b', alpha = 0.5)
plt.plot(np.median(np.median(random_ablation_grid_cosine_sim_1_step, axis = 1), axis = 0), 'bo-', label = 'Random')
plt.legend()
if save_flag:
    plt.savefig(save_path + '1_step_autocorrelation_vs_ablations_' + options.trajectory_style + '.png')
    plt.savefig(save_path + '1_step_autocorrelation_vs_ablations_' + options.trajectory_style + '.svg', format = 'svg')

#fig = plt.figure(figsize = (6, 4))
#plt.fill_between(np.arange(0, max_shift, 1), np.percentile(base_grid_cosine_sim, 25, axis = 0), np.percentile(base_grid_cosine_sim, 75, axis = 0), color = 'k', alpha = 0.3)
#plt.plot(np.median(base_grid_cosine_sim, axis = 0), 'k-', label = 'Base')
#colors = ['g', 'r', 'b']
#for ii in range(n_optimal_shifts):
    #plt.fill_between(np.arange(0, max_shift, 1), np.percentile(predictive_optimal_shift_ablation_grid_cosine_sim[:, ii, :], 25, axis = 0), np.percentile(predictive_optimal_shift_ablation_grid_cosine_sim[:, ii, :], 75, axis = 0), color = 'g', alpha = 0.5)
#    plt.plot(np.median(predictive_optimal_shift_ablation_grid_cosine_sim[:, ii, :], axis = 0), '-', color=colors[ii], label = 'Predictive: shift ' + str(0.5 * (optimal_shifts_ablate_min[ii] + optimal_shifts_ablate_max[ii])))
#plt.legend()
#plt.axis([0, 30, 0.4, 1.0])
#if save_flag:
#    plt.savefig(save_path + 'autocorrelation_vs_ablations_' + options.trajectory_style + '.png')
#    plt.savefig(save_path + 'autocorrelation_vs_ablations_' + options.trajectory_style + '.svg', format = 'svg')

# Running stats 
print('Predictive vs base:')
print(scipy.stats.ks_2samp(predictive_ablation_grid_cosine_sim, base_grid_cosine_sim, alternative = 'less').pvalue)

print('Retrospective vs base:')
print(scipy.stats.ks_2samp(retrospective_ablation_grid_cosine_sim, base_grid_cosine_sim, alternative = 'less').pvalue)

print('Predictive vs band:')
print(scipy.stats.ks_2samp(band_ablation_grid_cosine_sim, base_grid_cosine_sim, alternative = 'less').pvalue)

print('Predictive vs random:')
print(scipy.stats.ks_2samp(np.median(random_ablation_grid_cosine_sim, axis = 1), base_grid_cosine_sim, alternative = 'less').pvalu)



