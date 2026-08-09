#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:36:05 2026

@author: wredman
"""

"""Band-cell analysis based on NeurIPS 2024 band-score definition.

This script computes band scores from rate maps, compares them with
predictive/retrospective/normal grid classes, and generates summary figures.

Outputs are written under analysis_outputs/<model>/<seed>/<run>/band_cells/ (or
analysis_outputs/<seed>/band_cells/ if checkpoints are outside Models/).
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from path_utils import analysis_dir_for_checkpoint
from visualize import compute_ratemaps
from scores import band_scores, border_score
# %%
from multi_seed_predictive_analysis import build_options, infer_dims_from_state, expand_checkpoints



COLORS = {
    "predictive": "#1f77b4",
    "retrospective": "#d62728",
    "normal": "#7f7f7f",
    "low_grid": "#bdbdbd",
    "band": "#2ca02c",
    "border": "#ff7f0e",
}


def extract_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def load_gridness_data(ckpt_path: str) -> Dict[str, np.ndarray]:
    out_dir = analysis_dir_for_checkpoint(Path(ckpt_path))
    grid_path = out_dir / "gridness_data.npz"
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"Missing gridness_data.npz beside checkpoint: {grid_path}. "
            "Run multi_seed_predictive_analysis.py first."
        )
    with np.load(grid_path) as data:
        return {k: data[k] for k in data.files}


def safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.full(arr.shape[1], -1, dtype=int)
    vals = np.full(arr.shape[1], np.nan, dtype=float)
    for u in range(arr.shape[1]):
        col = arr[:, u]
        if not np.isfinite(col).any():
            continue
        i = int(np.nanargmax(col))
        idxs[u] = i
        vals[u] = col[i]
    return idxs, vals


def select_band_units(scores: np.ndarray, percentile: float, threshold: Optional[float]) -> Tuple[np.ndarray, float]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([], dtype=int), float("nan")
    if threshold is not None:
        cutoff = float(threshold)
    else:
        cutoff = float(np.nanpercentile(finite, percentile))
    return np.where(scores >= cutoff)[0], cutoff

def analyse_checkpoint(ckpt_path: str, args) -> Dict[str, float]:
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    ckpt_dir = os.path.dirname(ckpt_path)
    out_dir = os.path.join(ckpt_dir, "analysis_outputs", "border_band")
    os.makedirs(out_dir, exist_ok=True)

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    Ng_eval = min(args.Ng_use, Ng)
    idxs = np.arange(Ng_eval, dtype=int)
    activations, ratemap, _, _ = compute_ratemaps(
        model,
        traj_gen,
        options,
        res=args.res,
        n_avg=args.n_avg,
        Ng=Ng_eval,
        idxs=idxs,
    )
    band_vals, band_kx, band_ky = band_scores(activations, args.res, options.box_width,
                                              k_values=np.arange(args.band_k_min, args.band_k_max + 1e-6, args.band_k_step))

    border_vals = np.zeros((Ng_eval,), dtype=np.float32)
    for i in range(Ng_eval):
        bscore, _, _ = border_score(activations[i], args.res, options.box_width)
        border_vals[i] = bscore

    band_units, band_cutoff = select_band_units(band_vals, args.band_percentile, args.band_threshold)
    band_mask = np.zeros((Ng_eval,), dtype=bool)
    band_mask[band_units] = True

    border_mask = border_vals >= args.border_threshold

    summary = {
        "checkpoint": ckpt_path,
        "num_units_scored": int(Ng_eval),
        "band": {
            "percentile": float(args.band_percentile),
            "threshold": None if args.band_threshold is None else float(args.band_threshold),
            "cutoff": float(band_cutoff),
            "count": int(band_units.size),
            "fraction": float(band_units.size / max(Ng_eval, 1)),
        },
        "border": {
            "threshold": float(args.border_threshold),
            "count": int(np.sum(border_mask)),
        }}
        

    with open(out_dir + "/band_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir + "/band_scores.npz",
        band_scores=band_vals,
        band_kx=band_kx,
        band_ky=band_ky,
        border_scores=border_vals,
        band_units=band_units,
    )

    return {
        "band_count": int(band_units.size),
        "border_count": int(np.sum(border_mask)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", default = ["/Users/wredman/Documents/GitHub/predictive-grid-cells/RNNs/models/random_walk/Seed 9 weight decay 1e-04/steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/final_model.pth"], help="Path to a trained model (.pth).")  #'Straight/Seed 0 weight decay 1e-04/steps_20_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_straight/final_model.pth", help="Path to a trained model (.pth).")
    parser.add_argument("--trajectory_style", default="random_walk")
    
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--sequence_length", default=40, type=int)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--res", type=int, default=20)
    parser.add_argument("--n_avg", default=None, type=int)
    parser.add_argument("--n_batches", default=10, type=int)
    parser.add_argument("--Ng_use", type=int, default=4096)
    parser.add_argument("--band_percentile", type=float, default=90.0)
    parser.add_argument("--band_threshold", type=float, default=None)
    parser.add_argument("--band_k_max", type=float, default=1.5)
    parser.add_argument("--band_k_min", type=float, default=0.3)
    parser.add_argument("--band_k_step", type=float, default=0.05)
    parser.add_argument("--border_threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trajectory_fixed_speed", default=None)
    parser.add_argument("--trajectory_dt", default=0.02, type = float)
    parser.add_argument("--traj_speed_scale", default=1.0, type=float)
    parser.add_argument("--traj_speed_max", default=None, type=float)
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float)
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float)
    parser.add_argument("--traj_border_region", default=0.03, type=float)
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float)
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = args.checkpoint_paths
    if not checkpoints:
        raise FileNotFoundError("No checkpoints matched the provided paths.")
    for ckpt in checkpoints:
        print(f"[band_cell_analysis] Processing {ckpt}")
        analyse_checkpoint(ckpt, args)


if __name__ == "__main__":
    main()