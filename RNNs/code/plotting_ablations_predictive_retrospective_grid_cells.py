#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:08:41 2026

@author: wredman
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy

# Global 
ablation_type = 'ablations_fixed_number'
if ablation_type == 'ablations_fixed_number':
    x = [30, 60, 90, 120, 150]
else:
    x = [0.1, 0.2, 0.3, 0.4, 0.5]

shift_mode = 'temporal'

# Loading data data
data_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/' + ablation_type + '/' + shift_mode + ' shift/'

predictive_ablation = np.load(data_path + 'predictive_ablations.npy')
retrospective_ablation = np.load(data_path + 'retrospective_ablations.npy')
grid_ablation = np.load(data_path + 'grid_ablations.npy')
if ablation_type == 'ablations_fixed_number':
    predictive_ablation_baseline = np.load(data_path + 'random_ablations.npy')
    retrospective_ablation_baseline = np.load(data_path + 'random_ablations.npy')
    grid_ablation_baseline = np.load(data_path + 'random_ablations.npy')
else:
    predictive_ablation_baseline = np.load(data_path + 'random_predictive_baseline_ablations.npy')
    retrospective_ablation_baseline = np.load(data_path + 'random_retrospective_baseline_ablations.npy')
    grid_ablation_baseline = np.load(data_path + 'random_grid_baseline_ablations.npy')

# Plotting 
save_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/results/results/' + ablation_type + '/' + shift_mode + ' shift/'
save_flag = False

median_grid_ablation = np.median(grid_ablation, axis = 2)
median_grid_ablation_baseline = np.median(grid_ablation_baseline, axis = 2)

plt.figure()
plt.fill_between(x, np.percentile(median_grid_ablation, 25, axis = 0), np.percentile(median_grid_ablation, 75, axis = 0), color = 'k', alpha = 0.5)
plt.plot(x, np.median(median_grid_ablation, axis = 0), 'k-', label = 'Grid')
plt.fill_between(x, np.percentile(median_grid_ablation_baseline, 25, axis = 0), np.percentile(median_grid_ablation_baseline, 75, axis = 0), color = 'r', alpha = 0.5)
plt.plot(x, np.median(median_grid_ablation_baseline, axis = 0), 'r-', label = 'Random')
if ablation_type == 'ablations_fixed_number':
    plt.xlabel('# ablated units')
else:
    plt.xlabel('Ablation percentile')
plt.ylabel('Median decoding error')
plt.legend()
if save_flag:
    plt.savefig(save_path + 'grid_cell_ablations.png')
    plt.savefig(save_path + 'grid_cell_ablations.svg', format = 'svg')

print(scipy.stats.ks_2samp(median_grid_ablation , median_grid_ablation_baseline, alternative = 'less').pvalue)

median_predictive_ablation = np.median(predictive_ablation, axis = 2)
median_predictive_ablation_baseline = np.median(predictive_ablation_baseline, axis = 2)

plt.figure()
plt.fill_between(x, np.percentile(median_predictive_ablation, 25, axis = 0), np.percentile(median_predictive_ablation, 75, axis = 0), color = 'k', alpha = 0.5)
plt.plot(x, np.median(median_predictive_ablation, axis = 0), 'k-', label = 'Predictive')
plt.fill_between(x, np.percentile(median_predictive_ablation_baseline, 25, axis = 0), np.percentile(median_predictive_ablation_baseline, 75, axis = 0), color = 'r', alpha = 0.5)
plt.plot(x, np.median(median_predictive_ablation_baseline, axis = 0), 'r-', label = 'Random')
if ablation_type == 'ablations_fixed_number':
    plt.xlabel('# ablated units')
else:
    plt.xlabel('Ablation percentile')
plt.ylabel('Median decoding error')
plt.legend()
if save_flag:
    plt.savefig(save_path + 'predictive_grid_cell_ablations.png')
    plt.savefig(save_path + 'predictive_grid_cell_ablations.svg', format = 'svg')

print(scipy.stats.ks_2samp(median_predictive_ablation , median_predictive_ablation_baseline, alternative = 'less').pvalue)

median_retrospective_ablation = np.median(retrospective_ablation, axis = 2)
median_retrospective_ablation_baseline = np.median(retrospective_ablation_baseline, axis = 2)

plt.figure()
plt.fill_between(x, np.percentile(median_retrospective_ablation, 25, axis = 0), np.percentile(median_retrospective_ablation, 75, axis = 0), color = 'k', alpha = 0.5)
plt.plot(x, np.median(median_retrospective_ablation, axis = 0), 'k-', label = 'Retrospective')
plt.fill_between(x, np.percentile(median_retrospective_ablation_baseline, 25, axis = 0), np.percentile(median_retrospective_ablation_baseline, 75, axis = 0), color = 'r', alpha = 0.5)
plt.plot(x, np.median(median_retrospective_ablation_baseline, axis = 0), 'r-', label = 'Random')
if ablation_type == 'ablations_fixed_number':
    plt.xlabel('# ablated units')
else:
    plt.xlabel('Ablation percentile')
plt.ylabel('Median decoding error')
plt.legend()
if save_flag:
    plt.savefig(save_path + 'retrospective_grid_cell_ablations.png')
    plt.savefig(save_path + 'retrospective_grid_cell_ablations.svg', format = 'svg')

print(scipy.stats.ks_2samp(median_retrospective_ablation , median_retrospective_ablation_baseline, alternative = 'less').pvalue)


