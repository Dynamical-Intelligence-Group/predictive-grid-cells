#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 19:43:10 2026

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
shift_mode = 'temporal'
if shift_mode == 'temporal':
    shifts = np.arange(-20, 21, 1)
else:
    shifts = np.arange(-0.4, 0.4, 0.02)
n_shifts = len(shifts)
band_border_thresh_percentile = 95

grid_units = np.zeros((n_seeds, n_units))
predictive_units = np.zeros((n_seeds, n_units))
retrospective_units = np.zeros((n_seeds, n_units))
dead_units = np.zeros((n_seeds, n_units))
grid_scores = np.zeros((n_seeds, n_units))
max_shifted_grid_scores = np.zeros((n_seeds, n_units))
best_shift_ids = np.zeros((n_seeds, n_units))
shifted_grid_scores = np.zeros((n_seeds, n_units, n_shifts))
band_scores = np.zeros((n_seeds, n_units))
border_scores = np.zeros((n_seeds, n_units))
ratemaps = np.zeros((n_seeds, n_units, n_shifts, res, res))

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/' 

for ss in range(n_seeds):
    # Loading normal, predictive, and retrospective grid cell data
    grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/' + shift_mode + ' shift/predictive_retrospective/'          
    X = np.load(grid_cell_analysis_path + 'final_model.pth_' + trajectory_style + '_summary_data.npz')
    max_shifted_grid_scores[ss, :] = X['best_scores']
    shifted_grid_scores[ss, :, :] = X['scores_60'].T
    best_shift_ids[ss, :] = X['best_idx']
    
    shifted_ratemaps = X['shifted_ratemap']
    shifts = X['lag']
    grid_scores[ss, :] =  X['scores_60'][np.abs(shifts) < 1e-06, :].flatten()
    ratemaps[ss, :, :, :, :] = np.transpose(shifted_ratemaps, (3, 0, 1, 2))
    
    grid_ids = np.load(grid_cell_analysis_path + 'grid_ids_' + trajectory_style + '.npy')
    predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + trajectory_style + '.npy')
    retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + trajectory_style + '.npy')
    dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + trajectory_style + '.npy')
    grid_units[ss, grid_ids] = 1
    predictive_units[ss, predictive_ids] = 1
    retrospective_units[ss, retrospective_ids] = 1
    dead_units[ss, dead_unit_ids] = 1

    # Loading border and band cell data
    border_band_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(ss) + ' weight decay 1e-04/' + folder_name + 'analysis_outputs/border_band/'     
    X = np.load(border_band_cell_analysis_path + 'band_scores.npz')
    band_scores[ss, :] = X['band_scores']
    band_scores[ss, np.isnan(band_scores[ss, :])] = 0
    border_scores[ss, :] = X['border_scores']
    border_scores[ss, np.isnan(border_scores[ss, :])] = 0

# Plotting border and band score distribution
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/border_band/' + shift_mode + ' shift/'

plt.figure(figsize = (6, 4))
plt.hist(border_scores.flatten(), bins = np.arange(-1.0, 1.1, 0.2))
plt.xlabel('Border scores')
if fig_save:
    plt.savefig(save_path + 'Border_score_distribution.png')
    plt.savefig(save_path + 'Border_score_distribution.svg', format = 'svg')
plt.show()

plt.figure(figsize = (6, 4))
plt.hist(band_scores.flatten(), bins = np.arange(0.0, 1.1, 0.1))
plt.xlabel('Band scores')
if fig_save:
    plt.savefig(save_path + 'Band_score_distribution.png')
    plt.savefig(save_path + 'Band_score_distribution.svg', format = 'svg')
plt.show()

# Classifying border and band units
border_units = np.zeros((n_seeds, n_units))
band_units = np.zeros((n_seeds, n_units))

for ss in range(n_seeds):
    band_thresh = np.percentile(band_scores[ss, :], band_border_thresh_percentile )
    border_thresh = np.percentile(border_scores[ss, :], band_border_thresh_percentile )
    
    border_ids = np.argwhere(border_scores[ss, :] > border_thresh)
    border_units[ss, border_ids] = 1 
    border_units[ss, dead_units[ss, :] == 1] = 0
                                 
    band_ids = np.argwhere(band_scores[ss, :] > band_thresh)
    band_units[ss, band_ids] = 1 
    band_units[ss, dead_units[ss, :] == 1] = 0
    
if fig_save: 
    np.save(save_path + 'band_units.npy', band_units)
    np.save(save_path + 'border_units.npy', border_units)
    
# Plotting all border and band cells
plot_all_border = False
seed_plot = 0
if plot_all_border: 
    border_unit_ids = np.argwhere(border_units[seed_plot, :] == 1)
    for ii in range(len(border_unit_ids)):
        plt.figure(figsize = (4, 4))
        plt.title('Unit ' + str(border_unit_ids[ii]))
        plt.axis('off')
        plt.imshow(np.squeeze(rgb(ratemaps[seed_plot, border_unit_ids[ii], np.argwhere(np.abs(shifts) < 1e-6), :, :])))
        plt.show() 
        
        input("Press Enter to continue...") # Pause for user input in the console
        print("User pressed Enter, program continues.")
        plt.close() # Close the figure
        
plot_all_band = False
seed_plot = 0
if plot_all_band: 
    band_unit_ids = np.argwhere(band_units[seed_plot, :] == 1)
    for ii in range(len(band_unit_ids)):
        plt.figure(figsize = (4, 4))
        plt.title('Unit ' + str(band_unit_ids[ii]))
        plt.axis('off')
        plt.imshow(np.squeeze(rgb(ratemaps[seed_plot, band_unit_ids[ii], np.argwhere(np.abs(shifts) < 1e-6), :, :])))
        plt.show() 
        
        input("Press Enter to continue...") # Pause for user input in the console
        print("User pressed Enter, program continues.")
        plt.close() # Close the figure
    
# Identifying border/band overlap with grid/predictive/retrospective
band_grid_overlap = band_units * grid_units
band_predictive_overlap = band_units * predictive_units
band_retrospective_overlap = band_units * retrospective_units 
percent_band_grid_overlap = np.sum(band_grid_overlap, axis = 1) / np.sum(band_units, axis = 1)
percent_band_predictive_overlap = np.sum(band_predictive_overlap, axis = 1) / np.sum(band_units, axis = 1)
percent_band_retrospective_overlap = np.sum(band_retrospective_overlap, axis = 1) / np.sum(band_units, axis = 1)

border_grid_overlap = border_units * grid_units 
border_predictive_overlap = border_units * predictive_units
border_retrospective_overlap = border_units * retrospective_units 
percent_border_grid_overlap = np.sum(border_grid_overlap, axis = 1) / np.sum(border_units, axis = 1)
percent_border_predictive_overlap = np.sum(border_predictive_overlap, axis = 1) / np.sum(border_units, axis = 1)
percent_border_retrospective_overlap = np.sum(border_retrospective_overlap, axis = 1) / np.sum(border_units, axis = 1)

# Plotting example overlapping border/band and predictive/retrospective cells
n_plot = 12
seed_plot = 1
border_predictive_ids = np.argwhere(border_predictive_overlap[seed_plot, :] == 1)
border_predictive_ids = border_predictive_ids[np.argsort(border_scores[seed_plot, border_predictive_ids][:, 0])]
border_retrospective_ids = np.argwhere(border_retrospective_overlap[seed_plot, :] == 1)
border_retrospective_ids = border_retrospective_ids[np.argsort(border_scores[seed_plot, border_retrospective_ids][:, 0])]

band_predictive_ids = np.argwhere(band_predictive_overlap[seed_plot, :] == 1)
band_predictive_ids = band_predictive_ids[np.argsort(band_scores[seed_plot, band_predictive_ids][:, 0])]
band_retrospective_ids = np.argwhere(band_retrospective_overlap[seed_plot, :] == 1)
band_retrospective_ids = band_retrospective_ids[np.argsort(band_scores[seed_plot, band_retrospective_ids][:, 0])]

plt.figure(figsize = (12, 6))
plt.suptitle('Example border-predictive cells')
for ii in range(n_plot):
    if ii < len(border_predictive_ids):
        plt.subplot(2, n_plot, ii + 1) 
        plt.imshow(rgb(ratemaps[seed_plot, border_predictive_ids[-ii], np.argwhere(np.abs(shifts) < 1e-6), :, :])[0][0])
        plt.title('$\Delta$ = 0')
for ii in range(n_plot):
    if ii < len(border_predictive_ids):
        plt.subplot(2, n_plot, ii + 1 + n_plot) 
        plt.imshow(rgb(ratemaps[seed_plot, border_predictive_ids[-ii], int(best_shift_ids[seed_plot, border_predictive_ids[-ii]]),  :, :][0]))
        plt.title('$\Delta$ = ' + str(shifts[int(best_shift_ids[seed_plot, border_predictive_ids[-ii]])]))
    
plt.figure(figsize = (8, 6))
plt.suptitle('Example border-retrospective cells')
for ii in range(n_plot):
    if ii < len(border_retrospective_ids):
        plt.subplot(2, n_plot, ii + 1) 
        plt.imshow(rgb(ratemaps[seed_plot, border_retrospective_ids[-ii], np.argwhere(np.abs(shifts) < 1e-6), :, :])[0][0])
        plt.title('$\Delta$ = 0')
for ii in range(n_plot):
    if ii < len(border_retrospective_ids):
        plt.subplot(2, n_plot, ii + 1 + n_plot) 
        plt.imshow(rgb(ratemaps[seed_plot, border_retrospective_ids[-ii], int(best_shift_ids[seed_plot, border_retrospective_ids[-ii]]),  :, :][0]))
        plt.title('$\Delta$ = ' + str(shifts[int(best_shift_ids[seed_plot, border_retrospective_ids[-ii]])]))
    
plt.figure(figsize = (8, 6))
plt.suptitle('Example band-predictive cells')
for ii in range(n_plot):
    plt.subplot(2, n_plot, ii + 1) 
    plt.imshow(rgb(ratemaps[seed_plot, band_predictive_ids[-ii], np.argwhere(np.abs(shifts) < 1e-6), :, :])[0][0])
    plt.title('$\Delta$ = 0')
for ii in range(n_plot):
    plt.subplot(2, n_plot, ii + 1 + n_plot) 
    plt.imshow(rgb(ratemaps[seed_plot, band_predictive_ids[-ii], int(best_shift_ids[seed_plot, band_predictive_ids[-ii]]),  :, :][0]))
    plt.title('$\Delta$ = ' + str(shifts[int(best_shift_ids[seed_plot, band_predictive_ids[-ii]])]))
if fig_save:
    plt.savefig(save_path + 'Example_band_predictive_units.png')
    plt.savefig(save_path + 'Example_band_predictive_units.svg', format = 'svg')
plt.show()    

plt.figure(figsize = (8, 6))
plt.suptitle('Example band-retrospective cells')
for ii in range(n_plot):
    plt.subplot(2, n_plot, ii + 1) 
    plt.imshow(rgb(ratemaps[seed_plot, band_retrospective_ids[-ii], np.argwhere(np.abs(shifts) < 1e-6), :, :])[0][0])
    plt.title('$\Delta$ = 0')
for ii in range(n_plot):
    plt.subplot(2, n_plot, ii + 1 + n_plot) 
    plt.imshow(rgb(ratemaps[seed_plot, band_retrospective_ids[-ii], int(best_shift_ids[seed_plot, band_retrospective_ids[-ii]]),  :, :][0]))
    plt.title('$\Delta$ = ' + str(shifts[int(best_shift_ids[seed_plot, band_retrospective_ids[-ii]])]))
if fig_save:
    plt.savefig(save_path + 'Example_band_retrospective_units.png')
    plt.savefig(save_path + 'Example_band_retrospective_units.svg', format = 'svg')
plt.show()

# Plotting grid/predictive/retrospective and border/band overlaps
grid_chance_overlap = np.sum(grid_units, axis = 1) / (n_units - np.sum(dead_units, axis = 1))
plt.figure(figsize = (6, 4))
plt.plot(['Border-grid', 'Band-grid', 'Chance'], [percent_border_grid_overlap, percent_band_grid_overlap, grid_chance_overlap], 'ko')
plt.plot(['Border-grid', 'Band-grid', 'Chance'], [np.nanmedian(percent_border_grid_overlap), np.nanmedian(percent_band_grid_overlap), np.nanmedian(grid_chance_overlap)], 'rs')
plt.plot(['Border-grid', 'Border-grid'], [np.percentile(percent_border_grid_overlap, 25), np.percentile(percent_border_grid_overlap, 75)], 'r-')
plt.plot(['Band-grid', 'Band-grid'], [np.percentile(percent_band_grid_overlap, 25), np.percentile(percent_band_grid_overlap, 75)], 'r-')
plt.plot(['Chance', 'Chance'], [np.percentile(grid_chance_overlap, 25), np.percentile(grid_chance_overlap, 75)], 'r-')
plt.ylabel('Overlap (%)')
if fig_save:
    plt.savefig(save_path + 'Grid_cell_band_border_overlap.png')
    plt.savefig(save_path + 'Grid_cell_band_border_overlap.svg', format = 'svg')
plt.show()
print(scipy.stats.ks_2samp(percent_border_grid_overlap, grid_chance_overlap, alternative = 'less').pvalue)
print(scipy.stats.ks_2samp(percent_band_grid_overlap, grid_chance_overlap, alternative = 'less').pvalue)

predictive_chance_overlap = np.sum(predictive_units, axis = 1) / (n_units - np.sum(dead_units, axis = 1))
plt.figure(figsize = (6, 4))
plt.plot(['Border-predictive', 'Band-predictive', 'Chance'], [percent_border_predictive_overlap, percent_band_predictive_overlap, predictive_chance_overlap], 'ko')
plt.plot(['Border-predictive', 'Band-predictive', 'Chance'], [np.nanmedian(percent_border_predictive_overlap), np.nanmedian(percent_band_predictive_overlap), np.nanmedian(predictive_chance_overlap)], 'rs')
plt.plot(['Border-predictive', 'Border-predictive'], [np.percentile(percent_border_predictive_overlap, 25), np.percentile(percent_border_predictive_overlap, 75)], 'r-')
plt.plot(['Band-predictive', 'Band-predictive'], [np.percentile(percent_band_predictive_overlap, 25), np.percentile(percent_band_predictive_overlap, 75)], 'r-')
plt.plot(['Chance', 'Chance'], [np.percentile(predictive_chance_overlap, 25), np.percentile(predictive_chance_overlap, 75)], 'r-')
plt.ylabel('Overlap (%)')
if fig_save:
    plt.savefig(save_path + 'Predictive_grid_cell_band_border_overlap.png')
    plt.savefig(save_path + 'Predictive_grid_cell_band_border_overlap.svg', format = 'svg')
plt.show()
print(scipy.stats.ks_2samp(percent_border_predictive_overlap, predictive_chance_overlap, alternative = 'less').pvalue)
print(scipy.stats.ks_2samp(percent_band_predictive_overlap, predictive_chance_overlap, alternative = 'less').pvalue)

retrospective_chance_overlap = np.sum(retrospective_units, axis = 1) / (n_units - np.sum(dead_units, axis = 1))
plt.figure(figsize = (6, 4))
plt.plot(['Border-retrospective', 'Band-retrospective', 'Chance'], [percent_border_retrospective_overlap, percent_band_retrospective_overlap, retrospective_chance_overlap], 'ko')
plt.plot(['Border-retrospective', 'Band-retrospective', 'Chance'], [np.nanmedian(percent_border_retrospective_overlap), np.nanmedian(percent_band_retrospective_overlap), np.nanmedian(retrospective_chance_overlap)], 'rs')
plt.plot(['Border-retrospective', 'Border-retrospective'], [np.percentile(percent_border_retrospective_overlap, 25), np.percentile(percent_border_retrospective_overlap, 75)], 'r-')
plt.plot(['Band-retrospective', 'Band-retrospective'], [np.percentile(percent_band_retrospective_overlap, 25), np.percentile(percent_band_retrospective_overlap, 75)], 'r-')
plt.plot(['Chance', 'Chance'], [np.percentile(retrospective_chance_overlap, 25), np.percentile(retrospective_chance_overlap, 75)], 'r-')
plt.ylabel('Overlap (%)')
if fig_save:
    plt.savefig(save_path + 'Retrospective_grid_cell_band_border_overlap.png')
    plt.savefig(save_path + 'Retrospective_grid_cell_band_border_overlap.svg', format = 'svg')
plt.show()
print(scipy.stats.ks_2samp(percent_border_retrospective_overlap, retrospective_chance_overlap, alternative = 'less').pvalue)
print(scipy.stats.ks_2samp(percent_band_retrospective_overlap, retrospective_chance_overlap, alternative = 'less').pvalue)

# Plotting predictive/retrospective grid score increase for border/band units
median_border_predictive_grid_increase = np.zeros(n_seeds)
all_border_predictive_grid_increase = []
median_border_retrospective_grid_increase = np.zeros(n_seeds)
all_border_retrospective_grid_increase = []
for ss in range(n_seeds):
    max_predictive_grid_score = np.max(shifted_grid_scores[ss, border_units[ss, :] == 1, np.argwhere(shifts > 0)], axis = 0)
    median_border_predictive_grid_increase[ss] = np.nanmedian((max_predictive_grid_score - grid_scores[ss, border_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, border_units[ss, :] == 1]))
    all_border_predictive_grid_increase.append((max_predictive_grid_score - grid_scores[ss, border_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, border_units[ss, :] == 1]))
    
    max_retrospective_grid_score = np.max(shifted_grid_scores[ss, border_units[ss, :] == 1, np.argwhere(shifts < 0)], axis = 0)
    median_border_retrospective_grid_increase[ss] = np.nanmedian((max_retrospective_grid_score - grid_scores[ss, border_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, border_units[ss, :] == 1]))
    all_border_retrospective_grid_increase.append((max_retrospective_grid_score - grid_scores[ss, border_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, border_units[ss, :] == 1]))
    
plt.figure(figsize = (6, 4))
plt.plot(['Predictive', 'Retrospective'], [median_border_predictive_grid_increase, median_border_retrospective_grid_increase], 'ko')
plt.plot(['Predictive', 'Retrospective'], [np.nanmedian(median_border_predictive_grid_increase), np.nanmedian(median_border_retrospective_grid_increase)], 'rs')
plt.plot(['Predictive', 'Predictive'], [np.percentile(median_border_predictive_grid_increase, 25), np.percentile(median_border_predictive_grid_increase, 75)] ,'r-')
plt.plot(['Retrospective', 'Retrospective'], [np.percentile(median_border_retrospective_grid_increase, 25), np.percentile(median_border_retrospective_grid_increase, 75)] ,'r-')
plt.ylabel('Relative grid score increase')  
plt.title('Border units')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_border_units.png')
    plt.savefig(save_path + 'Relative_grid_score_border_units.svg', format = 'svg')  
plt.show()
print(scipy.stats.ks_2samp(median_border_predictive_grid_increase, median_border_retrospective_grid_increase, alternative = 'greater').pvalue)

plt.figure(figsize = (6, 4))
plt.hist(np.concatenate(all_border_predictive_grid_increase), bins = np.arange(-0.2, 1.2, 0.1), density = True, alpha = 0.5, label = 'Predictive')
plt.hist(np.concatenate(all_border_retrospective_grid_increase), bins = np.arange(-0.2, 1.2, 0.1), density = True, alpha = 0.5, label = 'Retrospective')
plt.legend()
plt.xlabel('Relative grid increase')
plt.title('Border units')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_increase_all_border_cells.png')
    plt.savefig(save_path + 'Relative_grid_score_increase_all_border_cells.svg', format = 'svg')
plt.show()
print('% border cells with predictive rel grid increase > 0.1:' + str(np.sum(np.concatenate(all_border_predictive_grid_increase) > 0.1) / np.sum(border_units)))
print('% border cells with retrospective rel grid increase > 0.1:' + str(np.sum(np.concatenate(all_border_retrospective_grid_increase) > 0.1) / np.sum(border_units)))

median_band_predictive_grid_increase = np.zeros(n_seeds)
median_band_non_grid_predictive_grid_increase = np.zeros(n_seeds)
all_band_predictive_grid_increase = []
all_band_band_scores = []
median_band_retrospective_grid_increase = np.zeros(n_seeds)
all_band_retrospective_grid_increase = []
for ss in range(n_seeds):
    max_predictive_grid_score = np.max(shifted_grid_scores[ss, band_units[ss, :] == 1, np.argwhere(shifts > 0)], axis = 0)
    median_band_predictive_grid_increase[ss] = np.nanmedian((max_predictive_grid_score - grid_scores[ss, band_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, band_units[ss, :] == 1]))
    all_band_predictive_grid_increase.append((max_predictive_grid_score - grid_scores[ss, band_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, band_units[ss, :] == 1]))
    
    max_predictive_grid_score_non_grid = np.max(shifted_grid_scores[ss, (band_units[ss, :] == 1) & (grid_units[ss, :] == 0), int(np.ceil(n_shifts / 2)):], axis = 1)
    median_band_non_grid_predictive_grid_increase[ss] = np.nanmedian((max_predictive_grid_score_non_grid - grid_scores[ss, (band_units[ss, :] == 1) & (grid_units[ss, :] == 0)]) / np.abs(grid_scores[ss, (band_units[ss, :] == 1) & (grid_units[ss, :] == 0)]))

    all_band_band_scores.append(band_scores[ss, band_units[ss, :] == 1])

    max_retrospective_grid_score = np.max(shifted_grid_scores[ss, band_units[ss, :] == 1, np.argwhere(shifts < 0)], axis = 0)
    median_band_retrospective_grid_increase[ss] = np.nanmedian((max_retrospective_grid_score - grid_scores[ss, band_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, band_units[ss, :] == 1]))
    all_band_retrospective_grid_increase.append((max_retrospective_grid_score - grid_scores[ss, band_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, band_units[ss, :] == 1]))

plt.figure(figsize = (6, 4))
plt.plot(['Predictive', 'Retrospective'], [median_band_predictive_grid_increase, median_band_retrospective_grid_increase], 'ko')
plt.plot(['Predictive', 'Retrospective'], [np.nanmedian(median_band_predictive_grid_increase), np.nanmedian(median_band_retrospective_grid_increase)], 'rs')
plt.plot(['Predictive', 'Predictive'], [np.percentile(median_band_predictive_grid_increase, 25), np.percentile(median_band_predictive_grid_increase, 75)] ,'r-')
plt.plot(['Retrospective', 'Retrospective'], [np.percentile(median_band_retrospective_grid_increase, 25), np.percentile(median_band_retrospective_grid_increase, 75)] ,'r-')
plt.ylabel('Relative grid score increase')  
plt.title('Band units')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_band_units.png')
    plt.savefig(save_path + 'Relative_grid_score_band_units.svg', format = 'svg')
plt.show()
print(scipy.stats.ks_2samp(median_band_predictive_grid_increase, median_band_retrospective_grid_increase, alternative = 'less').pvalue)

plt.figure(figsize = (6, 4))
plt.hist(np.concatenate(all_band_predictive_grid_increase), bins = np.arange(-0.2, 1.2, 0.1), density = True, alpha = 0.5, label = 'Predictive')
plt.hist(np.concatenate(all_band_retrospective_grid_increase), bins = np.arange(-0.2, 1.2, 0.1), density = True, alpha = 0.5, label = 'Retrospective')
plt.legend()
plt.xlabel('Relative grid increase')
plt.title('Band units')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_increase_all_band_cells.png')
    plt.savefig(save_path + 'Relative_grid_score_increase_all_band_cells.svg', format = 'svg')
plt.show()
print('% band cells with predictive rel grid increase > 0.1:' + str(np.sum(np.concatenate(all_band_predictive_grid_increase) > 0.1) / np.sum(band_units)))
print('% band cells with retrospective rel grid increase > 0.1:' + str(np.sum(np.concatenate(all_band_retrospective_grid_increase) > 0.1) / np.sum(band_units)))

median_grid_predictive_grid_increase = np.zeros(n_seeds)
all_grid_predictive_grid_increase = []
median_grid_retrospective_grid_increase = np.zeros(n_seeds)
all_grid_retrospective_grid_increase = []
for ss in range(n_seeds):
    max_predictive_grid_score = np.max(shifted_grid_scores[ss, grid_units[ss, :] == 1, np.argwhere(shifts > 0)], axis = 0)
    median_grid_predictive_grid_increase[ss] = np.nanmedian((max_predictive_grid_score - grid_scores[ss, grid_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, grid_units[ss, :] == 1]))
    all_grid_predictive_grid_increase.append((max_predictive_grid_score - grid_scores[ss, grid_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, grid_units[ss, :] == 1]))
    
    max_retrospective_grid_score = np.max(shifted_grid_scores[ss, grid_units[ss, :] == 1, np.argwhere(shifts < 0)], axis = 0)
    median_grid_retrospective_grid_increase[ss] = np.nanmedian((max_retrospective_grid_score - grid_scores[ss, grid_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, grid_units[ss, :] == 1]))
    all_grid_retrospective_grid_increase.append((max_retrospective_grid_score - grid_scores[ss, grid_units[ss, :] == 1 ]) / np.abs(grid_scores[ss, grid_units[ss, :] == 1]))
    
plt.figure(figsize = (6, 4))
plt.plot(['Predictive', 'Retrospective'], [median_grid_predictive_grid_increase, median_grid_retrospective_grid_increase], 'ko')
plt.plot(['Predictive', 'Retrospective'], [np.nanmedian(median_grid_predictive_grid_increase), np.nanmedian(median_grid_retrospective_grid_increase)], 'rs')
plt.plot(['Predictive', 'Predictive'], [np.percentile(median_grid_predictive_grid_increase, 25), np.percentile(median_grid_predictive_grid_increase, 75)] ,'r-')
plt.plot(['Retrospective', 'Retrospective'], [np.percentile(median_grid_retrospective_grid_increase, 25), np.percentile(median_grid_retrospective_grid_increase, 75)] ,'r-')
plt.ylabel('Relative grid score increase')  
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_grid_units.png')
    plt.savefig(save_path + 'Relative_grid_score_grid_units.svg', format = 'svg')
plt.show()
print(scipy.stats.ks_2samp(median_grid_predictive_grid_increase, median_grid_retrospective_grid_increase, alternative = 'greater').pvalue)

plt.figure(figsize = (6, 4))
plt.hist(np.hstack(all_grid_predictive_grid_increase), bins = np.arange(-0.2, 1.2, 0.1), density = True, alpha = 0.5, label = 'Predictive')
plt.hist(np.hstack(all_grid_retrospective_grid_increase), bins = np.arange(-0.2, 1.2, 0.1), density = True, alpha = 0.5, label = 'Retrospective')
plt.legend()
plt.xlabel('Relative grid increase')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_increase_all_grid_cells.png')
    plt.savefig(save_path + 'Relative_grid_score_increase_all_grid_cells.svg', format = 'svg')
plt.show()
print('% grid cells with predictive rel grid increase > 0.1:' + str(np.sum(np.array(np.hstack(all_grid_predictive_grid_increase)) > 0.1) / np.sum(grid_units)))
print('% grid cells with retrospective rel grid increase > 0.1:' + str(np.sum(np.array(np.hstack(all_grid_retrospective_grid_increase)) > 0.1) / np.sum(grid_units)))

plt.figure(figsize = (6, 4))
plt.plot(['Predictive-band','Predictive-grid', 'Retrospective-band', 'Retrospective-grid'], [median_band_predictive_grid_increase, median_grid_predictive_grid_increase, median_band_retrospective_grid_increase, median_grid_retrospective_grid_increase], 'ko')
plt.plot(['Predictive-band','Predictive-grid', 'Retrospective-band', 'Retrospective-grid'], [np.nanmedian(median_band_predictive_grid_increase), np.nanmedian(median_grid_predictive_grid_increase), np.nanmedian(median_band_retrospective_grid_increase), np.nanmedian(median_grid_retrospective_grid_increase)], 'rs')
plt.plot(['Predictive-band', 'Predictive-band'], [np.percentile(median_band_predictive_grid_increase, 25), np.percentile(median_band_predictive_grid_increase, 75)] ,'r-')
plt.plot(['Retrospective-band', 'Retrospective-band'], [np.percentile(median_band_retrospective_grid_increase, 25), np.percentile(median_band_retrospective_grid_increase, 75)] ,'r-')
plt.plot(['Predictive-grid', 'Predictive-grid'], [np.percentile(median_grid_predictive_grid_increase, 25), np.percentile(median_grid_predictive_grid_increase, 75)] ,'r-')
plt.plot(['Retrospective-grid', 'Retrospective-grid'], [np.percentile(median_grid_retrospective_grid_increase, 25), np.percentile(median_grid_retrospective_grid_increase, 75)] ,'r-')
plt.ylabel('Relative grid score increase')  
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_band_vs_grid_units.png')
    plt.savefig(save_path + 'Relative_grid_score_band_vs_grid_units.svg', format = 'svg')
plt.show()

print(scipy.stats.ks_2samp(median_band_predictive_grid_increase, median_grid_predictive_grid_increase, alternative = 'less').pvalue)
print(scipy.stats.ks_2samp(median_band_retrospective_grid_increase, median_grid_retrospective_grid_increase, alternative = 'less').pvalue)

all_band_predictive_grid_increase = np.concatenate(all_band_predictive_grid_increase)
all_band_predictive_grid_increase[all_band_predictive_grid_increase > 1.0] = 1.1
all_border_predictive_grid_increase = np.concatenate(all_border_predictive_grid_increase)
all_border_predictive_grid_increase[all_border_predictive_grid_increase > 1.0] = 1.1
all_grid_predictive_grid_increase = np.concatenate(all_grid_predictive_grid_increase)
all_grid_predictive_grid_increase[all_grid_predictive_grid_increase > 1.0] = 1.1
plt.figure(figsize = (6, 4))
plt.hist(all_band_predictive_grid_increase, bins = np.linspace(-0.2, 1.2, 8), density = True, alpha = 0.5, label = 'Band')
plt.hist(all_grid_predictive_grid_increase, bins = np.linspace(-0.2, 1.2, 8), density = True, alpha = 0.5, label = 'Grid')
plt.legend()
plt.xlabel('Relative grid increase')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_increase_all_grid_all_band.png')
    plt.savefig(save_path + 'Relative_grid_score_increase_all_grid_all_band.svg', format = 'svg')
plt.show()


plt.figure(figsize = (6, 4))
plt.hist(all_border_predictive_grid_increase, bins = np.linspace(-0.2, 1.2, 8), density = True, alpha = 0.5, label = 'Border')
plt.hist(all_grid_predictive_grid_increase, bins = np.linspace(-0.2, 1.2, 8), density = True, alpha = 0.5, label = 'Grid')
plt.legend()
plt.xlabel('Relative grid increase')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_increase_all_grid_all_border.png')
    plt.savefig(save_path + 'Relative_grid_score_increase_all_grid_all_border.svg', format = 'svg')
plt.show()


plt.figure(figsize = (6, 4))
plt.hist(all_band_predictive_grid_increase, bins = np.linspace(-0.2, 1.2, 8), density = True, alpha = 0.5, label = 'Band')
plt.hist(all_border_predictive_grid_increase, bins = np.linspace(-0.2, 1.2, 8), density = True, alpha = 0.5, label = 'Border')
plt.legend()
plt.xlabel('Relative grid increase')
if fig_save: 
    plt.savefig(save_path + 'Relative_grid_score_increase_all_band_all_border.png')
    plt.savefig(save_path + 'Relative_grid_score_increase_all_band_all_border.svg', format = 'svg')
plt.show()

print(scipy.stats.ks_2samp(all_band_predictive_grid_increase, all_border_predictive_grid_increase, alternative = 'greater').pvalue)
                   





