#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:42:57 2026

@author: wredman
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

import argparse
from utils import generate_run_ID
import random
from tqdm import tqdm
from visualize import compute_ratemaps
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import coo_matrix
from umap.umap_ import compute_membership_strengths, smooth_knn_dist


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
                    default=20,
                    help='number of trajectories per batch') #200
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
seed_plot = 0
n_units = options.Ng

folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/'

# Loading normal, predictive, and retrospective grid cell data
grid_cell_analysis_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(seed_plot) + ' weight decay 1e-04/' + folder_name + '/analysis_outputs/predictive_retrospective/'          
X = np.load(grid_cell_analysis_path + 'final_model.pth_' + options.trajectory_style + '_summary_data.npz')

grid_ids = np.load(grid_cell_analysis_path + 'grid_ids_' + options.trajectory_style + '.npy')
predictive_ids = np.load(grid_cell_analysis_path + 'predictive_ids_' + options.trajectory_style + '.npy')
retrospective_ids = np.load(grid_cell_analysis_path + 'retrospective_ids_' + options.trajectory_style + '.npy')
dead_unit_ids = np.load(grid_cell_analysis_path + 'dead_unit_ids_' + options.trajectory_style + '.npy')
non_dead_ids = np.arange(0, 4096, 1)[dead_unit_ids == False]
 
model_name = 'final_model.pth'
model_path = '/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed ' + str(seed_plot) + ' weight decay 1e-04/' + folder_name

# Computing low-dimensional transformation in un-ablated network
model = RNN(options, place_cells)
model = model.to(options.device) 
saved_model = torch.load(model_path + model_name, map_location=torch.device('cpu'))
 
res = 10
model.load_state_dict(saved_model)
traj_gen = TrajectoryGenerator(options, place_cells)
place_cells = PlaceCells(options)
rate_maps, _, g, _ = compute_ratemaps(model, traj_gen, options, res=res, t_start = 10)
rate_maps_flattened = rate_maps.reshape((n_units, res**2))
rate_maps_flattened = rate_maps_flattened[non_dead_ids, :].T
g = g[:, grid_ids[:, 0]]
#g = g[np.arange(0, np.shape(g)[0], 2), :]

pca_fit = PCA(n_components = 6).fit(g)
pca_result = pca_fit.transform(g)
umap_fit = umap.UMAP(n_components = 3, min_dist = 0.8, n_neighbors = 5000, metric = "cosine")
umap_result = umap_fit.fit_transform(pca_result)

# Plotting 
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

fig,axs = scatter3d(umap_result,pca_result[:, 0], nrows=2, ncols=2,azim_elev_title=False, 
                        s = 0.1, alpha = 0.5, dpi=300, figsize = (1,1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0.0, hspace = 0.0)



















