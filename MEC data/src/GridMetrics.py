"""
Grid cell metrics calculations: including grid score, orientation and spacing
"""
import os
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters
from skimage.feature import peak_local_max
from scipy.stats import circmean
from matplotlib.lines import Line2D
import random
import pickle
import json
from DataLoader import DataExtractor


def create_new_result_dir(results_working_directory, new_result_dir_name):
    """
    Creates a new directory for results if it does not already exist
    :param results_working_directory:
    :param new_result_dir_name:
    :return: new_result_dir
    """
    new_result_dir = os.path.join(results_working_directory, new_result_dir_name)
    if not os.path.exists(new_result_dir):
        print('Creating directory: ' + new_result_dir)
        os.makedirs(new_result_dir)
    else:
        print('Directory ' + new_result_dir + ' already exists')

    return new_result_dir


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """Calculating the grid scores with different radius."""
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)

    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x ** 2 + y ** 2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)


def get_even_odd_times(length_of_recording, seconds_per_bin=1, shuffled=True):
    """
    length_of_recording: int
            Length of recording variables vectors (bins)
    seconds_of_chunk: int
            Time in seconds that we want each odd/even chunk to be
    """

    length_of_chunk = int(100 * seconds_per_bin)  # Because every bin is 10ms [correct], 100 samples per second
    number_of_chunks = int(np.ceil(length_of_recording / length_of_chunk))
    half_length = int(np.ceil(number_of_chunks / 2))

    if shuffled:
        true_elements = [True] * half_length
        false_elements = [False] * half_length
        vector = true_elements + false_elements
        random.shuffle(vector)

        valid_times = np.array([[element] * length_of_chunk for element in vector]).flatten()

    else:
        valid_times = np.array([True, False] * int(np.ceil(length_of_recording / 2)))

    return valid_times[:length_of_recording]


def load_grid_metrics_from_pickle(rat, mod):
    general_results_working_directory = os.path.join(os.path.dirname(os.getcwd()), 'results')
    session_results_directory = create_new_result_dir(general_results_working_directory, rat + mod)
    filename = os.path.join(session_results_directory, rat + mod + '_grid_session.pickle')

    with open(filename, 'rb') as f:
        G = pickle.load(f)

    return G, general_results_working_directory, session_results_directory


def load_odd_even_metrics_from_json(rat, mod):
    general_results_working_directory = os.path.join(os.path.dirname(os.getcwd()), 'results')
    session_results_directory = create_new_result_dir(general_results_working_directory, rat + mod)
    filename = os.path.join(session_results_directory, rat + mod + '_odds-even.json')

    with open(filename, 'rb') as f:
        cell_trial_dict = json.load(f)

    cell_trial_dict = convert_keys_to_int(cell_trial_dict)

    return cell_trial_dict, general_results_working_directory, session_results_directory


def convert_keys_to_int(d: dict):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = convert_keys_to_int(v)
        new_dict[new_key] = v
    return new_dict


class GridScorer(object):
    """Class for scoring ratemaps given trajectories."""
#previously nbins = 50
    def __init__(self, x, y, spikes, nbins=50, coords_range=((-1, 1), (-1, 1))):
        """Scoring ratemaps given trajectories.
    Args:
      nbins: Number of bins per dimension in the ratemap.
      coords_range: Environment coordinates range.
      mask_parameters: parameters for the masks that analyze the angular
        autocorrelation of the 2D autocorrelation.
      min_max: Correction.
    """
        self._x = x
        self._y = y
        self._spikes = spikes
        self._nbins = nbins
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        # Create all masks
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        mask_parameters = zip(starts, ends)
        self._masks = [(self.get_ring_mask(mask_min, mask_max), (mask_min,
                                                                 mask_max))
                       for mask_min, mask_max in mask_parameters]
        # Mask for hiding the parts of the SAC that are never used
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan)

    def get_ring_mask(self, mask_min, mask_max, bins=None):
        if bins is None:
            bins = self._nbins

        n_points = [bins * 2 - 1, bins * 2 - 1]
        return (circle_mask(n_points, mask_max * bins) *
                (1 - circle_mask(n_points, mask_min * bins)))
    
    def _apply_delta_alignment(self, x, y, spikes, delta):
        """
        Shift the *alignment* between position and spikes by delta samples.
        delta > 0  means spikes occur after the position (retrospective).
        delta < 0  means spikes occur before the position (prospective).
        Returns arrays of equal length.
        """

        x = np.asarray(x); y = np.asarray(y); spikes = np.asarray(spikes)

        if delta == 0:
            n = min(len(x), len(y), len(spikes))
            return x[:n], y[:n], spikes[:n]
        
        T = min(len(x), len(y), len(spikes))
        d = int(np.clip(delta, -(T-1), T-1))
        
        if d > 0:
            # retrospective. neuron fires after position
            # line up spikes[d:] with x[:-d], y[:-d]
            return x[:-d], y[:-d], spikes[d:]
        else:
            k = -d
            # prospective. neuron fires before position
            # line up spikes[:-k] with x[k:], y[k:]
            return x[k:], y[k:], spikes[:-k]
        

    ###from scipy.ndimage import gaussian_filter
    #import numpy as np
    #import scipy.stats

    #def calculate_ratemap(self, cell, bins=None, delta=0):

        #if bins is None:
            #bins = self._nbins

        # Raw signals
        #x = self._x
        #y = self._y
        #spikes = self._spikes[cell]

        # Apply delta shift
        #x, y, spikes = self._apply_delta_alignment(x, y, spikes, int(delta))

        # --- Occupancy map (time spent per bin) ---
        #occupancy = scipy.stats.binned_statistic_2d(
            #x, y, None,
            #bins=bins,
            #statistic="count",
            #range=self._coords_range
        #)[0]

        # --- Spike count map ---
        #spike_map = scipy.stats.binned_statistic_2d(
            #x, y, spikes,
            #bins=bins,
            #statistic="sum",
            #range=self._coords_range
        #)[0]

        # --- Gaussian smoothing BEFORE division ---
        #sigma = 1.2  # tune if needed
        #occ_s = gaussian_filter(occupancy, sigma)
        #spike_s = gaussian_filter(spike_map, sigma)

        # --- Compute firing rate ---
        #with np.errstate(divide="ignore", invalid="ignore"):
            #ratemap = spike_s / occ_s

        # --- Mask low occupancy bins ---
        #valid = occ_s > 0
        #threshold = np.percentile(occ_s[valid], 5)
        #ratemap[occ_s < threshold] = np.nan

        #return ratemap

    def calculate_ratemap(self, cell, statistic='mean', bins=None, delta=0):
        ###"""Computes the ratemap given the coordinates and spikes.
        #Optional delta shifts the alignment between position and spikes by sample bins.
        #delta > 0. spikes after position. retrospective coding.
        #delta < 0. spikes before position. prospective coding.
        #delta = 0. no shift.
        ##"""
        if bins is None:
           bins = self._nbins

        # grab raw series
        x = self._x
        y = self._y
        spikes = self._spikes[cell]

        # apply alignment shift and trim to same length
        x, y, spikes = self._apply_delta_alignment(x, y, spikes, int(delta))

        # compute ratemap

        return scipy.stats.binned_statistic_2d(
            x, y, spikes,
            bins=bins,
            statistic=statistic,
            range=self._coords_range)[0]###
        
        
        # --- Occupancy map (time spent per bin) ---
        #occupancy = scipy.stats.binned_statistic_2d(
            #x, y, None,
            #bins=bins,
            #statistic='count',
            #range=self._coords_range
        #)[0]

        # --- Spike count map ---
        #spike_map = scipy.stats.binned_statistic_2d(
            #x, y, spikes,
            #bins=bins,
            #statistic='sum',
            #range=self._coords_range
        #)[0]

        # --- Smooth both ---
        #from scipy.ndimage import gaussian_filter

        #sigma = 1.2   # you may tune this
        #occ_s = gaussian_filter(occupancy, sigma)
        #spike_s = gaussian_filter(spike_map, sigma)

        # --- Compute rate ---
        #with np.errstate(divide='ignore', invalid='ignore'):
           # ratemap = spike_s / occ_s

        # --- Mask low occupancy bins ---
        #threshold = np.nanpercentile(occ_s, 5)
        #ratemap[occ_s < threshold] = np.nan

        #return ratemap

        # --- Occupancy map ---
       
        # --- Smooth both ---
        #from scipy.ndimage import gaussian_filter
        #sigma = 0.8

        #occ_s = gaussian_filter(occupancy, sigma)
        #spike_s = gaussian_filter(spike_map, sigma)

        # --- Compute rate ---
        #with np.errstate(divide='ignore', invalid='ignore'):
            #ratemap = spike_map/occupancy #spike_s / occ_s

        #ratemap[occupancy == 0] = np.nan

        #ratemap[occ_s == 0] = np.nan



        # --- VERY light smoothing ---
        #from scipy.ndimage import gaussian_filter
        #ratemap = gaussian_filter(ratemap, sigma=0.6)


        #with np.errstate(divide='ignore', invalid='ignore'):
            #ratemap = spike_map / occupancy

        #ratemap[occupancy == 0] = np.nan

        #return ratemap

    def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Plot ratemaps."""
        if ax is None:
            ax = plt.gca()
        # Plot the ratemap
        ax.imshow(ratemap, interpolation='none', *args, **kwargs)
        # ax.pcolormesh(ratemap, *args, **kwargs)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    def calculate_sac(self, cell, bins=None, delta=0):
        """Calculating spatial autocorrelogram."""
        if bins is None:
            bins = self._nbins

        # get a ratemap using the requested shift
        delta = int(delta)
        ratemap = self.calculate_ratemap(cell=cell, bins=bins, delta=delta)

        seq1 = ratemap
        seq2 = ratemap

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode='full')

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        std_seq1 = np.power(
            np.subtract(
                np.divide(sum_seq1_sq, n_bins),
                (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
        std_seq2 = np.power(
            np.subtract(
                np.divide(sum_seq2_sq, n_bins),
                (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins),
            np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        x_coef = np.nan_to_num(x_coef)

        return x_coef

    def plot_sac(self, sac, mask_params=None, ax=None, title=None, *args,
                 **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Plot spatial autocorrelogram."""
        if ax is None:
            ax = plt.gca()
        # Plot the sac
        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, interpolation='none', *args, **kwargs)
        # ax.pcolormesh(useful_sac, *args, **kwargs)
        # Plot a ring for the adequate mask
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[0] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor='k'))
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[1] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor='k'))
        ax.axis('off')
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)

    def grid_score_60(self, corr):
        return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    def grid_score_90(self, corr):
        return corr[90] - (corr[45] + corr[135]) / 2

    def rotated_sacs(self, sac, angles):
        return [
            scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
            for angle in angles
        ]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        # Calculate dc on the ring area
        masked_sac_mean = np.sum(masked_sac) / ring_area
        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered ** 2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

    def get_scores(self, sac):
        """Get summary of scores for grid cells."""

        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        scores = [
            self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for mask, mask_params in self._masks  # pylint: disable=unused-variable
        ]
        scores_60, scores_90, variances = map(np.asarray, zip(*scores))  # pylint: disable=unused-variable
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)

        return (scores_60[max_60_ind], scores_90[max_90_ind],
                self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac, max_60_ind)

    def get_sac_interp(self, cell):
        """Get interpolated sac."""
        sac = self.calculate_sac(cell)
        xx = np.linspace(-1, 1, np.shape(sac)[0])
        yy = np.linspace(-1, 1, np.shape(sac)[1])
        return scipy.interpolate.RegularGridInterpolator((xx, yy), sac)

    def get_phi(self, cell, interp=None, spacing_values=np.arange(0.15, 0.5, 0.05)):
        """Get spacing of grid cell."""

        if interp is None:
            interp = self.get_sac_interp(cell)

        n_angles = 1000
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        radial_values = []
        for r in spacing_values:
            values = interp(np.array([r * np.sin(angles), r * np.cos(angles)]).T)
            radial_values.append(values)

        radial_values = np.mean(radial_values, axis=0)
        peaks, _ = scipy.signal.find_peaks(radial_values, distance=n_angles / 7)
        if peaks.size == 6:
            phi = angles[peaks][:3]
        else:
            phi = np.zeros((3,))
            # print no phi found and the cell number
            print('no 6 angles found for cell {}'.format(cell))

        return phi, radial_values

    def get_spacing(self, cell, interp=None, phi=None):
        """Get spacing of grid cell. If no phi is given, it will take the first phi"""

        # if both interp and phi are not given, calculate them
        if interp is None:
            interp = self.get_sac_interp(cell)

        if phi is None:
            phi, _ = self.get_phi(cell, interp)
            phi = phi[0]

        spacing_vec = np.linspace(0.2, 0.5, 1000)
        spacing_values = []
        for r in spacing_vec:
            value = interp(np.array([r * np.sin(phi), r * np.cos(phi)]).T)
            spacing_values.append(value)

        spacing_values = np.array(spacing_values)
        spacing_peaks, _ = scipy.signal.find_peaks(spacing_values[:, 0], prominence=0.05)

        if spacing_peaks.size == 0:
            spacing = 0
        else:
            spacing = 100 * 2 * spacing_vec[spacing_peaks][0]

        return spacing


class GridParameters(object):

    def __init__(self, rat, mod, exp = 'OF', delta=0, precompute=False):
        self.rat = rat
        self.mod = mod
        self.exp = exp
        self.delta = int(delta)

        self.x, self.y, self.spikes = self.get_data()
        self.n_neurons = len(self.spikes)

        self.Scorer = GridScorer(self.x, self.y, self.spikes)
        if precompute:
            self.grid_scores = self.compute_all_grid_scores()
            self.mask_radius = self.get_mask_radius(plot_result=False)
        else:
            self.grid_scores = None
            self.mask_radius = None

        self.orientation = []
        self.spacing = []

    def save_class(self, save_dir):
        """Save class as a pickle file."""
        name = self.rat + self.mod + '_grid_session.pickle'
        with open(os.path.join(save_dir, name), 'wb') as f:
            pickle.dump(self, f)

    def get_data(self):
        """Get data for a rat and a module."""
        data = DataExtractor(self.rat, self.mod, self.exp)
        x, y, spikes, _ = data.get_data()
        return x, y, spikes

    def compute_grid_score(self, sac):
        """Compute grid score for a sac."""
        return self.Scorer.get_scores(sac)[0]

    def compute_cell_grid_score(self, cell):
        """Compute grid score for a cell."""
        sac = self.Scorer.calculate_sac(cell, delta=self.delta)
        return self.compute_grid_score(sac)

    def compute_all_grid_scores(self):
        """Compute grid score for all cells."""
        grid_scores = []
        for cell in range(self.n_neurons):
            grid_scores.append(self.compute_cell_grid_score(cell))
        return grid_scores

    def get_mask_radius(self, plot_result=False):
        """Get center field radius for the module"""
        cell = np.argmax(self.grid_scores)
        interp = self.Scorer.get_sac_interp(cell)
        spacing_vec = np.linspace(0, 1, 100)

        spacing_values = []
        for r in spacing_vec:
            value = interp(np.array([0, r]).T)[0]
            spacing_values.append(value)

        spacing_values = np.array(spacing_values)
        spacing_peaks, _ = scipy.signal.find_peaks(-1 * spacing_values)

        mask_radius = spacing_vec[spacing_peaks][0]

        if plot_result:
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            fig.suptitle('Best Cell: {}, Grid Score = {:.2f}'.format(cell, self.grid_scores[cell]), fontsize=22)

            ax = axs[1]
            mask = self.Scorer.get_ring_mask(mask_radius, 1)
            sac = self.Scorer.calculate_sac(cell) * mask
           #Old Code masked_sac = np.ma.masked_where(sac == 0, sac)
            # Old Code cmap = matplotlib.colormaps['viridis']
            # Old Code cmap.set_bad(color='white')

            sac_for_plot = sac.copy() #New Code
            # remove center peak for scaling only
            cx, cy = np.array(sac_for_plot.shape) // 2
            sac_for_plot[cx-2:cx+3, cy-2:cy+3] = np.nan
            absmax = np.nanpercentile(np.abs(sac_for_plot), 99)

            sac_display = np.ma.masked_where(sac == 0, sac) #New Code

            ax.imshow(
                #Old Code sac,
                sac_display,
                cmap='RdBu_r',
                vmin=-absmax,
                vmax=absmax,
                interpolation="nearest" #New Code
            ) #New Code

            # Old Code ax.imshow(masked_sac, cmap=cmap)
            ax.set_aspect('equal', 'box')
            ax.axis('off')
            ax.set_title('Masked SAC', fontsize=18)

            ax = axs[0]
            ax.plot(spacing_vec, spacing_values, linewidth=2)
            ax.scatter(spacing_vec[spacing_peaks], spacing_values[spacing_peaks], color='r')
            ax.scatter(spacing_vec[spacing_peaks][0], spacing_values[spacing_peaks][0], color='g')
            ax.set_xlabel('Radius', fontsize=16)
            ax.set_ylabel('SAC Value', fontsize=16)
            # set axis square
            ax.set_aspect('equal', 'box')
            ax.set_title('SAC Value in radial direction', fontsize=18)
            # Remove top and right splines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # set ticks of fontsize 12
            ax.tick_params(axis='both', which='major', labelsize=12)

        return mask_radius

    def get_coordinates_of_peaks(self, sac, polar=True, sigma=1):
        """Get coordinates of peaks in SAC."""

        bins = int((np.shape(sac)[0] + 1) / 2)
        mask = self.Scorer.get_ring_mask(self.mask_radius, 1, bins=bins)
        masked_sac = sac * mask
        masked_sac = filters.gaussian(masked_sac, sigma=sigma)

        sac_max = ndimage.maximum_filter(masked_sac, size=1, mode='constant')
        coordinates = peak_local_max(sac_max, threshold_abs=0.0, min_distance=5)

        if polar is True:
            xx, yy = np.meshgrid(np.linspace(-1, 1, bins * 2 - 1), np.linspace(-1, 1, bins * 2 - 1))
            r = []
            th = []

            for coord in coordinates:
                r.append(np.sqrt(xx[coord[0], coord[1]] ** 2 + yy[coord[0], coord[1]] ** 2))
                th.append(np.arctan2(yy[coord[0], coord[1]], xx[coord[0], coord[1]]))
            return np.array(r), np.array(th)
        else:
            return coordinates

    def get_coordinates_first_3_peaks(self, sac, polar=True):

        r, th = self.get_coordinates_of_peaks(sac, polar=True)

        r_six = r[np.argsort(r)[:6]]
        th_six = th[np.argsort(r)[:6]]

        first_three_peaks_idxs = np.argsort(np.abs(th_six))[:3]
        r_three = r_six[first_three_peaks_idxs]
        th_three = th_six[first_three_peaks_idxs]
        r_three[1:] = r_three[np.argsort(th_three[1:])]
        th_three[1:] = np.sort(th_three[1:])
        # sort r_three in the same way

        if polar is False:
            x = []
            y = []
            for r, th in zip(r_three, th_three):
                x.append(r * np.cos(th))
                y.append(r * np.sin(th))
            return x, y
        else:
            return r_three, th_three

    @staticmethod
    def check_if_orientation_is_valid(th_three):

        is_valid = []
        if len(th_three) >= 3:

            for theta in th_three[1:]:
                if np.abs(theta - th_three[0]) < np.pi / 6:
                    is_valid.append(False)
                elif np.abs(theta - th_three[0]) > np.pi / 2:
                    is_valid.append(False)
                else:
                    is_valid.append(True)

            if np.abs(th_three[1] - th_three[2]) < np.pi / 2:
                is_valid.append(False)

        else:
            is_valid.append(False)

        return np.all(is_valid)

    @staticmethod
    def check_if_spacing_is_valid(r_three):

        is_valid = []
        for r in r_three[1:]:
            if r / r_three[0] < 0.5:
                is_valid.append(False)
            elif r / r_three[0] > 1.5:
                is_valid.append(False)
            else:
                is_valid.append(True)

        return np.all(is_valid)

    def get_spacing_orientation_and_score(self, sac, grid_threshold=0.5):

        r_three, th_three = self.get_coordinates_first_3_peaks(sac)
        is_spacing_valid = self.check_if_spacing_is_valid(r_three)
        is_orientation_valid = self.check_if_orientation_is_valid(th_three)

        grid_score = self.compute_grid_score(sac)

        is_valid = is_orientation_valid and is_spacing_valid and grid_score > grid_threshold

        if is_valid:
            orientation = circmean(th_three, high=np.pi, low=-np.pi) * 180 / np.pi
            spacing = np.mean(r_three)
            return 2 * 100 * spacing, orientation, grid_score
        else:
            # if not is_spacing_valid:
            #     # print('Cell is not valid because spacing is not valid')
            # if not is_orientation_valid:
            #     # print('Cell is not valid because orientation is not valid')

            return None, None, None

    def get_cell_spacing_and_orientation(self, cell, grid_threshold=0):
        sac = self.Scorer.calculate_sac(cell) * self.Scorer.get_ring_mask(self.mask_radius, 1)
        spacing, orientation, score = self.get_spacing_orientation_and_score(sac)

        if score:
            if score < grid_threshold:
                # print('Cell is not valid because grid score is too low')
                spacing = None
                orientation = None

        return spacing, orientation

    def get_odd_and_even_sacs(self, cell, valid_times=None, seconds_per_bin=5):

        if valid_times is None:
            valid_times = get_even_odd_times(len(self.x), seconds_per_bin)

        odd_times = valid_times
        even_times = ~odd_times

        x_odd = self.x[odd_times]
        y_odd = self.y[odd_times]

        x_even = self.x[even_times]
        y_even = self.y[even_times]

        spikes_odd = []
        spikes_even = []
        for i in range(self.n_neurons):
            spikes_odd.append(self.spikes[i][odd_times])
            spikes_even.append(self.spikes[i][even_times])

        Scorer_odd = GridScorer(x_odd, y_odd, spikes_odd)
        sac_odd = Scorer_odd.calculate_sac(cell) * self.Scorer.get_ring_mask(self.mask_radius, 1)

        Scorer_even = GridScorer(x_even, y_even, spikes_even)
        sac_even = Scorer_even.calculate_sac(cell) * self.Scorer.get_ring_mask(self.mask_radius, 1)

        return sac_odd, sac_even

    def get_cell_odd_and_even_spacing_and_orientation(self, cell, valid_times=None, seconds_per_bin=5):

        sac_odd, sac_even = self.get_odd_and_even_sacs(cell, valid_times, seconds_per_bin)

        spacing_odd, orientation_odd, grid_score_odd = self.get_spacing_orientation_and_score(sac_odd)
        spacing_even, orientation_even, grid_score_even = self.get_spacing_orientation_and_score(sac_even)

        metrics = {'spacing_odd': spacing_odd, 'orientation_odd': orientation_odd,
                   'spacing_even': spacing_even, 'orientation_even': orientation_even,
                   'grid_score_odd': grid_score_odd, 'grid_score_even': grid_score_even}

        return metrics

    def get_all_odd_and_even_spacings_and_orientations(self, seconds_per_bin=5):

        spacings_odd = []
        orientations_odd = []
        spacings_even = []
        orientations_even = []
        for cell in range(self.n_neurons):

            if self.orientation[cell] is None:
                spacing_odd = None
                orientation_odd = None
                spacing_even = None
                orientation_even = None
            else:
                spacing_odd, orientation_odd, spacing_even, orientation_even = self.get_cell_odd_and_even_spacing_and_orientation(
                    cell, seconds_per_bin)

            spacings_odd.append(spacing_odd)
            orientations_odd.append(orientation_odd)
            spacings_even.append(spacing_even)
            orientations_even.append(orientation_even)

        return spacings_odd, orientations_odd, spacings_even, orientations_even

    def get_all_orientations_and_spacings(self):

        orientations = []
        spacings = []
        print('Computing orientation of all cells')
        for cell in range(self.n_neurons):
            print('\r Cell ' + str(cell) + '/' + str(self.n_neurons), end="")
            spacing, orientation = self.get_cell_spacing_and_orientation(cell)
            orientations.append(orientation)
            spacings.append(spacing)

        self.orientation = orientations
        self.spacing = spacings

        count_none = self.orientation.count(None)
        count_non_none = len(self.orientation) - count_none

        print('Valid cells are {} out of {}'.format(count_non_none, self.n_neurons))

        return orientations, spacings

    def show_orientation_and_spacing(self, input_data):
        """
        :param input_data: can be a cell or a sac
        :return:
        """

        if isinstance(input_data, int):
            cell = input_data
            sac = self.Scorer.calculate_sac(cell) * self.Scorer.get_ring_mask(self.mask_radius, 1)
            grid_score = self.grid_scores[cell]
            orientation = self.orientation[cell]
            spacing = self.spacing[cell]

        elif isinstance(input_data, np.ndarray):
            sac = input_data
            cell = None
            spacing, orientation, grid_score = self.get_spacing_orientation_and_score(sac)
        else:
            raise Exception('Input data must be a cell or a sac')

        x_three, y_three = self.get_coordinates_first_3_peaks(sac, polar=False)

        xx, yy = np.meshgrid(np.linspace(-1, 1, 99), np.linspace(-1, 1, 99))

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        masked_sac = np.ma.masked_where(sac == 0, sac)
        cmap = matplotlib.colormaps['viridis']
        cmap.set_bad(color='white')

        ax.pcolor(xx, yy, masked_sac, cmap=cmap)
        for x, y in zip(x_three, y_three):
            ax.scatter(x, y, color='r')
            line = Line2D([0, x], [0, y], color='r')
            ax.add_line(line)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        # Figure title
        if cell is not None:
            fig.suptitle('Cell {}'.format(cell), fontsize=18)

        if orientation is None:
            fig.text(0.5, 0.9, 'No metrics found', horizontalalignment="center", fontsize=14)
        else:
            fig.text(0.5, 0.9, 'G-score: {:.2f}, Orientation: {:.2f}, Spacing: {:.2f}'
                     .format(grid_score, orientation, spacing), horizontalalignment="center",
                     fontsize=14)

        fig.text(0.5, 0.05, 'Rat ' + str(self.rat) + ', Module ' + str(self.mod), horizontalalignment="center",
                 fontsize=14)

    def show_odd_even_orientation_and_spacing(self, sac_odd, sac_even):
        """
        :param input_data: can be a cell or a sac
        :return:
        """

        sacs = [sac_odd, sac_even]
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        for i, sac in enumerate(sacs):
            spacing, orientation, grid_score = self.get_spacing_orientation_and_score(sac)
            x_three, y_three = self.get_coordinates_first_3_peaks(sac, polar=False)

            xx, yy = np.meshgrid(np.linspace(-1, 1, 99), np.linspace(-1, 1, 99))

            masked_sac = np.ma.masked_where(sac == 0, sac)
            cmap = matplotlib.colormaps['viridis']
            cmap.set_bad(color='white')

            ax = axes[i]
            ax.pcolor(xx, yy, masked_sac, cmap=cmap)
            for x, y in zip(x_three, y_three):
                ax.scatter(x, y, color='r')
                line = Line2D([0, x], [0, y], color='r')
                ax.add_line(line)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            if orientation is None:
                ax.text(0, 0.9, 'No metrics found', horizontalalignment="center", fontsize=14)
            else:
                ax.text(0, -1
                        , r'G: {:.2f}, $\theta$: {:.2f}, $\lambda$: {:.2f}'
                        .format(grid_score, orientation, spacing), horizontalalignment="center",
                        fontsize=14)

    def show_coordinates_of_peaks(self, cell, sac=None, bins=50):

        if sac is None:
            sac = self.Scorer.calculate_sac(cell, bins=bins)
        else:
            bins = int((np.shape(sac)[0] + 1) / 2)

        mask = self.Scorer.get_ring_mask(self.mask_radius, 1, bins=bins)
        masked_sac = sac * mask
        masked_sac = np.ma.masked_where(masked_sac == 0, masked_sac)

        xx, yy = np.meshgrid(np.linspace(-1, 1, 2 * bins - 1), np.linspace(-1, 1, 2 * bins - 1))

        coordinates = self.get_coordinates_of_peaks(masked_sac, polar=False)
        x_three, y_three = self.get_coordinates_first_3_peaks(masked_sac, polar=False)

        # display results
        fig, ax = plt.subplots(1, 3, figsize=(14, 6))
        fig.suptitle('Cell {}'.format(cell), fontsize=22)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(wspace=0.05)

        cmap = matplotlib.colormaps['viridis']
        cmap.set_bad(color='white')

        ax[0].pcolor(xx, yy, masked_sac)
        ax[0].axis('off')
        ax[0].set_title('Original SAC', fontsize=18)

        ax[1].pcolor(xx, yy, masked_sac)
        for xy in coordinates:
            ax[1].scatter(xx[xy[0], xy[1]], yy[xy[0], xy[1]], color='r')
        ax[1].axis('off')
        ax[1].set_title('Coordinates of all peaks', fontsize=18)
        ax[1].autoscale(True)
        ax[1].set_aspect('equal')
        ax[1].set_xlim(-1, 1)
        ax[1].set_ylim(-1, 1)

        ax[2].pcolor(xx, yy, masked_sac)
        for x, y in zip(x_three, y_three):
            ax[2].scatter(x, y, color='r')
            line = Line2D([0, x], [0, y], color='r')
            ax[2].add_line(line)
        ax[2].set_xlim(-1, 1)
        ax[2].set_ylim(-1, 1)
        ax[2].set_title('First Three peaks', fontsize=18)
        ax[2].axis('off')

        for axis in ax.ravel():
            axis.set_aspect('equal')

        fig.tight_layout()

    def compute_session_odd_even_metrics(self, n_trials=3, seconds_per_bin=1):

        cell_trial_dict = {}

        valid_times_shuffled = []
        for trial in range(n_trials):
            valid_times_shuffled.append(get_even_odd_times(len(self.x), seconds_per_bin))

        valid_times_shuffled = np.array(valid_times_shuffled)

        for cell in range(self.n_neurons):

            if self.orientation[cell] is None:
                continue

            else:
                cell_trial_dict[cell] = {}
                for trial in range(n_trials):
                    print(
                        '\r Cell ' + str(cell + 1) + '/' + str(self.n_neurons) + ' Trial ' + str(trial + 1) + '/' + str(
                            n_trials), end="")
                    cell_trial_dict[cell][trial] = self.get_cell_odd_and_even_spacing_and_orientation(cell,
                                                                                                      valid_times_shuffled[
                                                                                                          trial],
                                                                                                      seconds_per_bin)

                    # if None in cell_trial_dict[cell][trial].values():
                    #     del cell_trial_dict[cell]
                    #     break

        return cell_trial_dict, valid_times_shuffled

    def compute_trial_odd_even_axis_metrics(self, cell, valid_time, upsample=False):

        sac_odd, sac_even = self.get_odd_and_even_sacs(cell, valid_time)
        _, _, grid_score_odd = self.get_spacing_orientation_and_score(sac_odd)
        _, _, grid_score_even = self.get_spacing_orientation_and_score(sac_even)

        if upsample:
            sac_odd = ndimage.zoom(sac_odd, 199 / 99)
            sac_even = ndimage.zoom(sac_even, 199 / 99)

            sac_odd = ndimage.gaussian_filter(sac_odd, 0.5)
            sac_even = ndimage.gaussian_filter(sac_even, 0.5)

        r_three_odd, th_three_odd = self.get_coordinates_first_3_peaks(sac_odd)
        r_three_odd = 2 * 100 * r_three_odd
        th_three_odd = th_three_odd * 180 / np.pi

        r_three_even, th_three_even = self.get_coordinates_first_3_peaks(sac_even)
        r_three_even = 2 * 100 * r_three_even
        th_three_even = th_three_even * 180 / np.pi

        metrics = {'r_odd': r_three_odd, 'th_odd': th_three_odd,
                   'r_even': r_three_even, 'th_even': th_three_even,
                   'score_odd': grid_score_odd, 'score_even': grid_score_even}

        return metrics

    def compute_session_odd_even_axis_metrics(self, valid_times, upsample=False):

        cell_trial_dict = {}

        for cell in range(self.n_neurons):

            if self.orientation[cell] is None:
                continue

            else:
                cell_trial_dict[cell] = {}
                for trial in range(len(valid_times)):
                    print('\r Cell ' + str(cell) + '/' + str(self.n_neurons) + ' Trial ' + str(trial + 1) + '/' + str(
                        len(valid_times)), end="")
                    cell_trial_dict[cell][trial] = self.compute_trial_odd_even_axis_metrics(cell, valid_times[trial],
                                                                                            upsample)

        return cell_trial_dict

    # def compute_session_odd_even_metrics_non_shuffled(self, n_trials, seconds_per_bin = 1):
    #
    #     cell_trial_dict = {}
    #     valid_times = get_even_odd_times(len(self.x), shuffled = False)
    #
    #     for cell in range(self.n_neurons):
    #
    #             if self.orientation[cell] is None:
    #                 continue
    #
    #             else:
    #                 cell_trial_dict[cell] = {}
    #                 for trial in range(n_trials):
    #                     print('\r Cell ' + str(cell + 1) + '/' + str(self.n_neurons) + ' Trial ' + str(trial + 1) + '/' + str(
    #                         n_trials), end="")
    #                     cell_trial_dict[cell][trial] = self.get_cell_odd_and_even_spacing_and_orientation(cell,
    #                                                                                                 valid_times,
    #                                                                                                 seconds_per_bin)
    #
    #                     if None in cell_trial_dict[cell][trial].values():
    #                         del cell_trial_dict[cell]
    #                         break
