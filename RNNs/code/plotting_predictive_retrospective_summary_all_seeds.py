#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 12:35:03 2026

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
trajectory_style = ['straight', 'random_walk']
shift_mode = 'spatial'
n_styles = len(trajectory_style)
shifts = np.arange(-20, 21, 1)
n_shifts = len(shifts)

grid_units = np.zeros((n_seeds, n_units, n_styles))
predictive_units = np.zeros((n_seeds, n_units, n_styles))
retrospective_units = np.zeros((n_seeds, n_units, n_styles))
dead_units = np.zeros((n_seeds, n_units, n_styles))
grid_scores = np.zeros((n_seeds, n_units, n_styles))
optimal_shifts = np.zeros((n_seeds, n_units, n_styles))
max_shifted_grid_scores = np.zeros((n_seeds, n_units, n_styles))

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

for tt in range(n_styles):
    for ss in range(n_seeds):
        # Loading normal, predictive, and retrospective grid cell data
        grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/' + shift_mode + ' shift/predictive_retrospective/'          
        X = np.load(grid_cell_analysis_path + 'final_model.pth_' + trajectory_style[tt] + '_summary_data.npz')
        grid_scores[ss, :, tt] =  X['zero_scores']
        max_shifted_grid_scores[ss, :, tt] = X['best_scores']
        optimal_shifts[ss, :, tt] = shifts[X['best_idx']]
        
        grid_ids = np.load(grid_cell_analysis_path + 'grid_ids_' + trajectory_style[tt] + '.npy')
        predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + trajectory_style[tt] + '.npy')
        retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + trajectory_style[tt] + '.npy')
        dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + trajectory_style[tt] + '.npy')
        grid_units[ss, grid_ids, tt] = 1
        predictive_units[ss, predictive_ids, tt] = 1
        retrospective_units[ss, retrospective_ids, tt] = 1
        dead_units[ss, dead_unit_ids, tt] = 1

# Plotting percent classification for all seeds
fig_save = False
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/grid_predictive_retrospective/' + shift_mode + ' shift/'

plt.figure(figsize = (6, 4))
plt.plot(['Grid', 'Predictive', 'Retrospective'], [np.sum(grid_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 
                                                   np.sum(predictive_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 
                                                   np.sum(retrospective_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1))], 'ko')
plt.plot(['Grid', 'Predictive', 'Retrospective'], [np.median(np.sum(grid_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1))), 
                                                  np.median(np.sum(predictive_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1))), 
                                                  np.median(np.sum(retrospective_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)))], 'rs')
plt.plot(['Grid', 'Grid'], [np.percentile(np.sum(grid_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 25), 
                            np.percentile(np.sum(grid_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 75)], 'r-')
plt.plot(['Predictive', 'Predictive'], [np.percentile(np.sum(predictive_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 25), 
                            np.percentile(np.sum(predictive_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 75)], 'r-')
plt.plot(['Retrospective', 'Retrospective'], [np.percentile(np.sum(predictive_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 25), 
                            np.percentile(np.sum(predictive_units[:, :, 0], axis = 1) / (n_units - np.sum(dead_units[:, :, 0], axis = 1)), 75)], 'r-')

plt.ylim(0, 0.8)
plt.ylabel('% classified')
if fig_save: 
    plt.savefig(save_path + 'grid_predictive_retrospective_classification_percent.png')
    plt.savefig(save_path + 'grid_predictive_retrospective_classification_percent.svg', format = 'svg')

# Plotting optimal shift for all classified units 
plt.figure(figsize = (6, 4))
plt.hist(optimal_shifts[grid_units[:, :, 0] == 1, 0].flatten())
plt.xlabel('Optimal shift')
plt.ylabel('Count')
plt.title('Grid cells')
if fig_save: 
    plt.savefig(save_path + 'optimal_shift_' + trajectory_style[0] + '_all_grid_units.png')
    plt.savefig(save_path + 'optimal_shift_' + trajectory_style[0] + '_all_grid_units.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.hist(optimal_shifts[predictive_units[:, :, 0] == 1, 0].flatten())
plt.xlabel('Optimal shift')
plt.ylabel('Count')
plt.title('Predictive cells')
if fig_save: 
    plt.savefig(save_path + 'optimal_shift_' + trajectory_style[0] + '_all_predictive_units.png')
    plt.savefig(save_path + 'optimal_shift_' + trajectory_style[0] + '_all_predictive_units.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.hist(optimal_shifts[retrospective_units[:, :, 0] == 1, 0].flatten())
plt.xlabel('Optimal shift')
plt.ylabel('Count')
plt.title('Retrospective cells')
if fig_save: 
    plt.savefig(save_path + 'optimal_shift_' + trajectory_style[0] + '_all_retrospective_units.png')
    plt.savefig(save_path + 'optimal_shift_' + trajectory_style[0] + '_all_retrospective_units.svg', format = 'svg')

# Plotting max grid score vs grid score at 0 ratio for predictive and retrospective grid units
gs_rel_inc_predictive = (max_shifted_grid_scores[predictive_units[:, :, 0] == 1, 0] - grid_scores[predictive_units[:, :, 0] == 1, 0]) / np.abs(grid_scores[predictive_units[:, :, 0] == 1, 0])
gs_rel_inc_predictive[gs_rel_inc_predictive > 1.0] = 1.1
plt.figure(figsize = (6, 4))
n_bins = 7
x_max = 1.2
plt.hist(gs_rel_inc_predictive, bins = np.linspace(0, x_max, n_bins))
plt.ylabel('# units')
plt.xlabel('Relative GS rel. inc.')
plt.title('Predictive grid unit GS rel. increase')
if fig_save: 
    plt.savefig(save_path + 'rel_GS_increase_all_predictive_' + trajectory_style[0] + '.png')
    plt.savefig(save_path + 'rel_GS_increase_all_predictive_' + trajectory_style[0] + '.svg', format = 'svg')

gs_rel_inc_retrospective = (max_shifted_grid_scores[retrospective_units[:, :, 0] == 1, 0] - grid_scores[retrospective_units[:, :, 0] == 1, 0]) / np.abs(grid_scores[retrospective_units[:, :, 0] == 1, 0])
gs_rel_inc_retrospective[gs_rel_inc_retrospective > 1.0] = 1.1
plt.figure(figsize = (6, 4))
plt.hist(gs_rel_inc_retrospective, bins = np.linspace(0, x_max, n_bins))
plt.ylabel('# units')
plt.xlabel('Relative GS improvement')
plt.title('Retrospective grid unit GS rel. inc.')
if fig_save: 
    plt.savefig(save_path + 'rel_GS_all_increase_retrospective_' + trajectory_style[0] +'.png')
    plt.savefig(save_path + 'rel_GS_all_increase_retrospective_' + trajectory_style[0] + '.svg', format = 'svg')

# Plotting average max score for the different trajectories
median_max_shifted_grid_scores_predictive = np.zeros((n_seeds, n_styles))
for nn in range(n_seeds):
    for tt in range(n_styles):
        median_max_shifted_grid_scores_predictive[nn, tt] = np.median(max_shifted_grid_scores[nn, predictive_units[nn, :, tt] == 1, tt])

plt.figure(figsize = (6, 4))
plt.plot(trajectory_style, median_max_shifted_grid_scores_predictive.T, 'ko')
plt.plot(trajectory_style, np.median(median_max_shifted_grid_scores_predictive, axis = 0), 'rs')
plt.plot([trajectory_style[0], trajectory_style[0]], [np.percentile(median_max_shifted_grid_scores_predictive[:, 0], 25), np.percentile(median_max_shifted_grid_scores_predictive[:, 0], 75)], 'r-')
plt.plot([trajectory_style[1], trajectory_style[1]], [np.percentile(median_max_shifted_grid_scores_predictive[:, 1], 25), np.percentile(median_max_shifted_grid_scores_predictive[:, 1], 75)], 'r-')
plt.ylabel('Max shifted grid score')
if fig_save: 
    plt.savefig(save_path + 'max_shifted_grid_score_vs_trajectory_style_predictive_grid_cell.png')
    plt.savefig(save_path + 'max_shifted_grid_score_vs_trajectory_style_predictive_grid_cell.svg', format = 'svg')

print(scipy.stats.ks_2samp(median_max_shifted_grid_scores_predictive[:, 0], median_max_shifted_grid_scores_predictive[:, 1], alternative = 'less').pvalue)

median_max_shifted_grid_scores_retrospective = np.zeros((n_seeds, n_styles))
for nn in range(n_seeds):
    for tt in range(n_styles):
        median_max_shifted_grid_scores_retrospective[nn, tt] = np.median(max_shifted_grid_scores[nn, retrospective_units[nn, :, tt] == 1, tt])

plt.figure(figsize = (6, 4))
plt.plot(trajectory_style, median_max_shifted_grid_scores_retrospective.T, 'ko')
plt.plot(trajectory_style, np.median(median_max_shifted_grid_scores_retrospective, axis = 0), 'rs')
plt.plot([trajectory_style[0], trajectory_style[0]], [np.percentile(median_max_shifted_grid_scores_retrospective[:, 0], 25), np.percentile(median_max_shifted_grid_scores_retrospective[:, 0], 75)], 'r-')
plt.plot([trajectory_style[1], trajectory_style[1]], [np.percentile(median_max_shifted_grid_scores_retrospective[:, 1], 25), np.percentile(median_max_shifted_grid_scores_retrospective[:, 1], 75)], 'r-')
plt.ylabel('Max shifted grid score')
if fig_save: 
    plt.savefig(save_path + 'max_shifted_grid_score_vs_trajectory_style_retrospective_grid_cell.png')
    plt.savefig(save_path + 'max_shifted_grid_score_vs_trajectory_style_retrospective_grid_cell.svg', format = 'svg')

print(scipy.stats.ks_2samp(median_max_shifted_grid_scores_retrospective[:, 0], median_max_shifted_grid_scores_retrospective[:, 1], alternative = 'less').pvalue)















