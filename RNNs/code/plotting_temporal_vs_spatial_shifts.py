#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:56:29 2026

@author: wredman
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt 
from visualize import rgb

# Globals 
fig_save = False

# Loading saved data

n_seeds = 10
n_units = 4096 
res = 20 
trajectory_style = 'random_walk'
save_flag = True

percent_spatial_predictive_units = np.zeros(n_seeds)
percent_spatial_retrospective_units = np.zeros(n_seeds)
max_spatial_shifted_grid_scores = []
percent_temporal_predictive_units = np.zeros(n_seeds)
percent_temporal_retrospective_units = np.zeros(n_seeds)
max_temporal_shifted_grid_scores = []

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

for ss in range(n_seeds):
    # Loading spatial shift
    shifts = np.arange(-0.4, 0.4, 0.02)
    grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/spatial shift/predictive_retrospective/'          
    X = np.load(grid_cell_analysis_path + 'final_model.pth_' + trajectory_style + '_summary_data.npz')
    grid_scores =  X['scores_60'][np.abs(shifts) < 1e-06, :].flatten()
    max_spatial_shifted_gs = X['best_scores']
        
    spatial_predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + trajectory_style + '.npy')
    spatial_retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + trajectory_style + '.npy')
    dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + trajectory_style + '.npy')
    percent_spatial_predictive_units[ss] = len(spatial_predictive_ids) / len(dead_unit_ids)
    percent_spatial_retrospective_units[ss] = len(spatial_retrospective_ids) / len(dead_unit_ids)
    
    # Loading temporal shift
    shifts = np.arange(-20, 21, 1)
    grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/temporal shift/predictive_retrospective/'          
    X = np.load(grid_cell_analysis_path + 'final_model.pth_' + trajectory_style + '_summary_data.npz')
    grid_scores =  X['scores_60'][np.abs(shifts) < 1e-06, :].flatten()
    max_temporal_shifted_gs = X['best_scores']
        
    temporal_predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + trajectory_style + '.npy')
    temporal_retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + trajectory_style + '.npy')
    dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + trajectory_style + '.npy')
    percent_temporal_predictive_units[ss] = len(temporal_predictive_ids) / len(dead_unit_ids)
    percent_temporal_retrospective_units[ss] = len(temporal_retrospective_ids) / len(dead_unit_ids)
    
    # 
    consistent_predictive_ids = np.intersect1d(spatial_predictive_ids, temporal_predictive_ids)
    max_spatial_shifted_grid_scores.append(max_spatial_shifted_gs[consistent_predictive_ids])
    max_temporal_shifted_grid_scores.append(max_temporal_shifted_gs[consistent_predictive_ids])   
        
        
# Plotting 
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/grid_predictive_retrospective/comparison/'

max_spatial_shifted_grid_scores = [item for sublist in max_spatial_shifted_grid_scores for item in sublist]
max_temporal_shifted_grid_scores = [item for sublist in max_temporal_shifted_grid_scores for item in sublist]

fig = plt.figure(figsize = (6, 4))
plt.plot([0.3, 1.3], [0.3, 1.3], 'r-')
plt.plot(max_spatial_shifted_grid_scores, max_temporal_shifted_grid_scores, 'ko')
plt.plot(np.median(max_spatial_shifted_grid_scores), np.median(max_temporal_shifted_grid_scores), 'r^')
plt.xlabel('GS(D$_max$)')
plt.ylabel('GS($\Delta_{max}$')
if save_flag: 
    plt.savefig(save_path + 'max_GS_spatial_vs_temporal_shift.png')
    plt.savefig(save_path + 'max_GS_spatial_vs_temporal_shift.svg', format = 'svg')

fig = plt.figure(figsize=(6, 4))        
plt.plot([0.0, 0.05], [0.0, 0.05], 'r-')
plt.plot(percent_spatial_predictive_units, percent_temporal_predictive_units, 'ko')
plt.xlabel('% pred. grid units spatial shift')
plt.ylabel('% pred. grid units temporal shift')
if save_flag:
    plt.savefig(save_path + 'percent_predictive_grid_units_spatial_vs_temporal_shift.png')
    plt.savefig(save_path + 'percent_predictive_grid_units_spatial_vs_temporal_shift.svg', format = 'svg')

fig = plt.figure(figsize=(6, 4))    
plt.plot([0.0, 0.05], [0.0, 0.05], 'r-')    
plt.plot(percent_spatial_retrospective_units, percent_temporal_retrospective_units, 'ko')
plt.xlabel('% retro. grid units spatial shift')
plt.ylabel('% retro. grid units temporal shift')
if save_flag:
    plt.savefig(save_path + 'percent_retrospective_grid_units_spatial_vs_temporal_shift.png')
    plt.savefig(save_path + 'percent_retrospective_grid_units_spatial_vs_temporal_shift.svg', format = 'svg')










        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        