#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:42:57 2026

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
                    default=100,
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

def cos_sim(X, Y):
    nX = np.linalg.norm(X, axis = 2)
    nY = np.linalg.norm(Y, axis = 2)
    
    c = np.sum(X * Y, axis = 2) / (nX * nY)
    
    return c

# Loading saved data
n_seeds = 10
n_random_ablations = 1
n_trajectories_plot = 1
shift_mode = 'temporal'
save_flag = False
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/ablations/'

band_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/border_band/' + shift_mode + ' shift/'     
band_units = np.load(band_cell_analysis_path + 'band_units.npy')

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

base_grid_euclidean_dist_norm = np.zeros((n_seeds, options.batch_size, options.sequence_length))
predictive_ablation_grid_euclidean_dist_norm = np.zeros((n_seeds, options.batch_size, options.sequence_length))
band_ablation_grid_euclidean_dist_norm = np.zeros((n_seeds, options.batch_size, options.sequence_length))
retrospective_ablation_grid_euclidean_dist_norm = np.zeros((n_seeds, options.batch_size, options.sequence_length))
random_ablation_grid_euclidean_dist_norm = np.zeros((n_seeds, options.batch_size, options.sequence_length, n_random_ablations))

base_grid_euclidean_dist_max = np.zeros((n_seeds, options.batch_size))
predictive_ablation_grid_euclidean_dist_max = np.zeros((n_seeds, options.batch_size))
band_ablation_grid_euclidean_dist_max = np.zeros((n_seeds, options.batch_size))
retrospective_ablation_grid_euclidean_dist_max= np.zeros((n_seeds, options.batch_size))
random_ablation_grid_euclidean_dist_max = np.zeros((n_seeds, options.batch_size, n_random_ablations))


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
        
        # Loading band units data
        band_ids = np.argwhere(band_units[ss, :] == 1)
        
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
        
        base_grid_activity = g[:, :, grid_ids]
        X_base = base_grid_activity.reshape((-1, base_grid_activity.shape[-2]))
        pca_fit = PCA(n_components = 6).fit(X_base)
        pca_fit_2d = PCA(n_components = 2).fit(X_base)
        pca_result = pca_fit.transform(X_base)
        pca_result_2d = pca_fit_2d.transform(X_base)
        print(np.sum(pca_fit_2d.explained_variance_))
        
        for bb in range(options.batch_size):
            individual_trajectory = base_grid_activity[:, bb, :, 0]
            starting_activity = np.repeat([individual_trajectory[0, :]], options.sequence_length, axis = 0)
            d = np.sqrt(np.sum((individual_trajectory - starting_activity [0, :])**2, axis = 1)) 
            base_grid_euclidean_dist_norm[ss, bb, :] = d / np.max(d)
            base_grid_euclidean_dist_max[ss, bb] = np.max(d)
        
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
                
        predictive_ablation_grid_activity = g[:, :, grid_ids]
        
        for bb in range(options.batch_size):
            individual_trajectory = predictive_ablation_grid_activity[:, bb, :, 0]
            starting_activity = np.repeat([individual_trajectory[0, :]], options.sequence_length, axis = 0)
            d = np.sqrt(np.sum((individual_trajectory - starting_activity [0, :])**2, axis = 1)) 
            predictive_ablation_grid_euclidean_dist_norm[ss, bb, :] = d / np.max(d)
            predictive_ablation_grid_euclidean_dist_max[ss, bb] = np.max(d)

        plt.figure(figsize = (4 * n_trajectories_plot, 4))
        for ii in range(n_trajectories_plot):
            pca_result_2d_trajectory = pca_fit_2d.transform(base_grid_activity[:, ii, :, 0])
            predictive_ablation_pca_result_2d = pca_fit_2d.transform(predictive_ablation_grid_activity[:, ii, :, 0])
            
            plt.subplot(1, n_trajectories_plot, ii + 1)
            plt.plot(pca_result_2d[:, 0], pca_result_2d[:, 1], 'ko', alpha = 0.3)
            plt.plot(pca_result_2d_trajectory[:, 0], pca_result_2d_trajectory[:, 1], 'wo-', label = 'base')
            plt.plot(predictive_ablation_pca_result_2d[:, 0], predictive_ablation_pca_result_2d[:, 1], 'go-', label = 'predictive ablation')
            plt.legend()
        plt.suptitle('Seed: ' + str(ss))
            
        if save_flag: 
            plt.savefig(save_path + 'predictive_ablations_seed_' + str(ss) + '.png')
            plt.savefig(save_path + 'predictive_ablations_seed_' + str(ss) + '.svg', format = 'svg')

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
        
        band_ablation_grid_activity = g[:, :, grid_ids]
        
        for bb in range(options.batch_size):
           individual_trajectory = band_ablation_grid_activity[:, bb, :, 0]
           starting_activity = np.repeat([individual_trajectory[0, :]], options.sequence_length, axis = 0)
           d = np.sqrt(np.sum((individual_trajectory - starting_activity [0, :])**2, axis = 1)) 
           band_ablation_grid_euclidean_dist_norm[ss, bb, :] = d / np.max(d)
           band_ablation_grid_euclidean_dist_max[ss, bb] = np.max(d)
            
        fig = plt.figure(figsize = (4 * n_trajectories_plot, 4))
        for ii in range(n_trajectories_plot):
            pca_result_2d_trajectory = pca_fit_2d.transform(base_grid_activity[:, ii, :, 0])
            band_ablation_pca_result_2d = pca_fit_2d.transform(band_ablation_grid_activity[:, ii, :, 0])
            
            plt.subplot(1, n_trajectories_plot, ii + 1)
            plt.plot(pca_result_2d[:, 0], pca_result_2d[:, 1], 'ko', alpha = 0.3)
            plt.plot(pca_result_2d_trajectory[:, 0], pca_result_2d_trajectory[:, 1], 'wo-', label = 'base')
            plt.plot(band_ablation_pca_result_2d[:, 0], band_ablation_pca_result_2d[:, 1], 'yo-', label = 'band ablation')
            plt.legend()
        plt.suptitle('Seed: ' + str(ss))
            
        if save_flag: 
            plt.savefig(save_path + 'band_ablations_seed_' + str(ss) + '.png')
            plt.savefig(save_path + 'band_ablations_seed_' + str(ss) + '.svg', format = 'svg')

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
        
        retrospective_ablation_grid_activity = g[:, :, grid_ids]
        
        for bb in range(options.batch_size):
            individual_trajectory = retrospective_ablation_grid_activity[:, bb, :, 0]
            starting_activity = np.repeat([individual_trajectory[0, :]], options.sequence_length, axis = 0)
            d = np.sqrt(np.sum((individual_trajectory - starting_activity [0, :])**2, axis = 1)) 
            retrospective_ablation_grid_euclidean_dist_norm[ss, bb, :] = d / np.max(d)
            retrospective_ablation_grid_euclidean_dist_max[ss, bb] = np.max(d)
            
        plt.figure(figsize = (4 * n_trajectories_plot, 4))
        for ii in range(n_trajectories_plot):
            pca_result_2d_trajectory = pca_fit_2d.transform(base_grid_activity[:, ii, :, 0])
            retrospective_ablation_pca_result_2d = pca_fit_2d.transform(retrospective_ablation_grid_activity[:, ii, :, 0])
            
            plt.subplot(1, n_trajectories_plot, ii + 1)
            plt.plot(pca_result_2d[:, 0], pca_result_2d[:, 1], 'ko', alpha = 0.3)
            plt.plot(pca_result_2d_trajectory[:, 0], pca_result_2d_trajectory[:, 1], 'wo-', label = 'base')
            plt.plot(retrospective_ablation_pca_result_2d[:, 0], retrospective_ablation_pca_result_2d[:, 1], 'ro-', label = 'retrospective ablation')
            plt.legend()
        plt.suptitle('Seed: ' + str(ss))
                       
        if save_flag: 
            plt.savefig(save_path + 'retrospective_ablations_seed_' + str(ss) + '.png')
            plt.savefig(save_path + 'retrospective_ablations_seed_' + str(ss) + '.svg', format = 'svg')
      
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
                  
           random_ablation_grid_activity = g[:, :, grid_ids]
           
           for bb in range(options.batch_size):
               individual_trajectory = random_ablation_grid_activity[:, bb, :, 0]
               starting_activity = np.repeat([individual_trajectory[0, :]], options.sequence_length, axis = 0)
               d = np.sqrt(np.sum((individual_trajectory - starting_activity [0, :])**2, axis = 1)) 
               random_ablation_grid_euclidean_dist_norm[ss, bb, :, aa] = d / np.max(d)
               random_ablation_grid_euclidean_dist_max[ss, bb, aa] = np.max(d)
      
# Plotting
plt.figure(figsize = (6, 4))
plt.fill_between(np.arange(0, options.sequence_length, 1), np.percentile(np.mean(base_grid_euclidean_dist_norm, axis = 1), 25, axis = 0), np.percentile(np.mean(base_grid_euclidean_dist_norm, axis = 1), 75, axis = 0), color = 'k', alpha = 0.5)
plt.plot(np.arange(0, options.sequence_length), np.median(np.mean(base_grid_euclidean_dist_norm, axis = 1), axis = 0), 'k-', label = 'Base')
plt.fill_between(np.arange(0, options.sequence_length, 1), np.percentile(np.mean(predictive_ablation_grid_euclidean_dist_norm, axis = 1), 25, axis = 0), np.percentile(np.mean(predictive_ablation_grid_euclidean_dist_norm, axis = 1), 75, axis = 0), color = 'g', alpha = 0.5)
plt.plot(np.arange(0, options.sequence_length), np.median(np.mean(predictive_ablation_grid_euclidean_dist_norm, axis = 1), axis = 0), 'g-', label = 'Predictive')
plt.fill_between(np.arange(0, options.sequence_length, 1), np.percentile(np.mean(retrospective_ablation_grid_euclidean_dist_norm, axis = 1), 25, axis = 0), np.percentile(np.mean(retrospective_ablation_grid_euclidean_dist_norm, axis = 1), 75, axis = 0), color = 'r', alpha = 0.5)
plt.plot(np.arange(0, options.sequence_length), np.median(np.mean(retrospective_ablation_grid_euclidean_dist_norm, axis = 1), axis = 0), 'r-', label = 'Retrospective')
plt.fill_between(np.arange(0, options.sequence_length, 1), np.percentile(np.mean(band_ablation_grid_euclidean_dist_norm, axis = 1), 25, axis = 0), np.percentile(np.mean(band_ablation_grid_euclidean_dist_norm, axis = 1), 75, axis = 0), color = 'y', alpha = 0.5)
plt.plot(np.arange(0, options.sequence_length), np.median(np.mean(band_ablation_grid_euclidean_dist_norm, axis = 1), axis = 0), 'y-', label = 'Band')
plt.fill_between(np.arange(0, options.sequence_length, 1), np.percentile(np.mean(np.mean(random_ablation_grid_euclidean_dist_norm, axis = 1), axis = 2), 25, axis = 0), np.percentile(np.mean(np.mean(random_ablation_grid_euclidean_dist_norm, axis = 1), axis = 2), 75, axis = 0), color = 'b', alpha = 0.5)
plt.plot(np.arange(0, options.sequence_length), np.median(np.mean(np.mean(random_ablation_grid_euclidean_dist_norm, axis = 1), axis = 2), axis = 0), 'b-', label = 'Random')
#plt.axis([35, 41, 0.75,0.95])

plt.figure(figsize = (6, 4))
plt.boxplot([np.mean(base_grid_euclidean_dist_max, axis = 1), np.mean(predictive_ablation_grid_euclidean_dist_max, axis = 1), np.mean(retrospective_ablation_grid_euclidean_dist_max, axis = 1), np.mean(band_ablation_grid_euclidean_dist_max, axis = 1), np.mean(np.mean(random_ablation_grid_euclidean_dist_max, axis = 2), axis = 1)], tick_labels = ['base', 'predictive', 'retrospective', 'band', 'random'] )
print(scipy.stats.ks_2samp(np.mean(np.mean(random_ablation_grid_euclidean_dist_max, axis = 2), axis = 1), np.mean(predictive_ablation_grid_euclidean_dist_max, axis = 1), alternative = 'greater').pvalue)






