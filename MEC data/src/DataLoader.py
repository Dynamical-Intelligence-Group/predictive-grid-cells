from typing import Dict

from numpy import load, ndarray
import numpy as np


def moser_data_extractor(rat, exp):
    """ Gets info for the Moser dataset"""

    if rat == 'q' and exp == 'OF':
        filename = '../data/rat_q_grid_modules_1_2.npz'
        start = 27826
        end = 31223
    elif rat == 'r1' and exp == 'OF':
        filename = '../data/rat_r_day1_grid_modules_1_2_3.npz'
        start = 7457
        end = 14778
    elif rat == 'r2' and exp == 'OF':
        filename = '../data/rat_r_day2_grid_modules_1_2_3.npz'
        start = 10617
        end = 13004
    elif rat == 'r2' and exp == 'sleep':
        filename = '../data/rat_r_day2_grid_modules_1_2_3.npz'
        start = 396
        end = 9941
    elif rat == 's' and exp == 'OF':
        filename = '../data/rat_s_grid_modules_1.npz'
        start = 9939
        end = 12363
    else:
        filename = None
        start = None
        end = None

    return filename, start, end


class DataExtractor(object):
    """
        This class extracts the binned coordinates and spikes from the Moser dataset.

        ...

        Attributes
        ----------
        filename : str
            a formatted string to print out what the animal says
        start : str
            the name of the animal
        end : str
            the sound that the animal makes
        x : array
            X location of the animal
        y: array
            Y location of the animal (normalized, we don't know real units)
        t: array
            Time-points (in seconds)
        spikes: array
            Amount of spikes per time bin

        mod: int
            Module that the recording is from

        Methods
        -------

        """

    def __init__(self, rat, mod, exp):

        filename, start, end = moser_data_extractor(rat, exp)

        self.filename = filename
        self.start = start
        self.end = end
        self.x = []
        self.y = []
        self.t = []
        self.spikes_raw = []
        self.spikes = []
        self.mod = mod

        self.extract_data()

    def load_data(self):
        f = load(self.filename, allow_pickle=True)
        self.x = f['x'][()]
        self.y = f['y'][()]
        self.t = f['t'][()]
        self.spikes_raw = f["spikes_mod" + self.mod][()]
        f.close()

    def crop_data(self):

        times = np.where((self.t >= self.start) & (self.t < self.end))
        self.x = self.x[times]
        self.y = self.y[times]
        self.t = self.t[times]

        cell_idx = np.arange(len(self.spikes_raw))
        spikes = {}
        for i, m in enumerate(cell_idx):
            s = self.spikes_raw[m]
            spikes[i] = np.array(s[(s >= self.start) & (s < self.end)])

        self.spikes_raw = spikes

    def bin_spikes(self):

        spikes_binned = {}
        for i in self.spikes_raw:
            spikes_digitized = np.digitize(self.spikes_raw[i], self.t[:-1], right=False)
            spk = np.zeros_like(self.t)
            for m in spikes_digitized:
                spk[m] += 1
            spikes_binned[i] = spk

        self.spikes = spikes_binned

    def extract_data(self):
        self.load_data()
        self.crop_data()
        self.bin_spikes()

    def get_data(self):
        return self.x, self.y, self.spikes, self.t
