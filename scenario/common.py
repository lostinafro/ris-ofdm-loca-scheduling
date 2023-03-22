# filename "common.py"
# Global methods: contains general methods used everywhere

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from datetime import date
import tikzplotlib
from scipy.constants import k

from tqdm import tqdm
import sys

# TODO: move the geometrical methods into a separate file that imports cupy

# Global dictionaries
cluster_shapes = {'box', 'circle', 'semicircle'}
channel_stats = {'LoS', 'fading'}
node_labels = {'BS': 0, 'UE': 1, 'RIS': 2}

# The following are defined for graphic purpose only
node_color = {'BS': '#DC2516',  'UE': '#36F507', 'RIS': '#0F4EEA'}
node_mark = {'BS': 'o', 'UE': 'x', 'RIS': '^'}

# The supported channel types are the following.
channel_types = {'LoS', 'No', 'AWGN', 'Rayleigh', 'Rice', 'Shadowing'}

# Custom distributions
def circular_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an annular region. The output is in polar coordinates.

    Parameters
    ----------
    :param n: int, number of points.
    :param r_outer: float, outer radius of the annular region.
    :param r_inner: float, inner radius of the annular region.
    :param rng: np.random.RandomState, random generator needed for reproducibility

    Returns
    -------
    rho: np.ndarray, distance of each point from center of the annular region.
    phi: np.ndarray, azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * rng.rand(n, 1)
    return np.hstack((rho, phi))


def semicircular_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an semi-annular region. The outputs is in polar coordinates.

    Parameters
    ----------
    :param n: int, number of points.
    :param r_outer: float, outer radius of the annular region.
    :param r_inner: float, inner radius of the annular region.
    :param rng: np.random.RandomState, random generator needed for reproducibility

    Returns
    -------
    rho: np.ndarray, distance of each point from center of the annular region.
    phi: np.ndarray, azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = np.pi * rng.rand(n, 1)
    return np.hstack((rho, phi))




# Physical noise
def thermal_noise(bandwidth, noise_figure=3, t0=293):
    """Compute the noise power [W] according to bandwidth and ambient temperature.

    :param bandwidth : float, receiver total bandwidth [Hz]
    :param noise_figure: float, noise figure of the receiver [dB]
    :param t0: float, ambient temperature [K]

    :return: power of the noise [W]
    """
    return k * bandwidth * t0 * db2lin(noise_figure)  # [W]


# Utilities
def lin2dB(lin):
    return 10 * np.log10(lin)

def db2lin(db):
    return 10 ** (db / 10)

def dbm2watt(dbm):
    """Simply converts dBm to Watt"""
    return 10 ** (dbm / 10 - 3)


def watt2dbm(watt):
    """Simply converts Watt to dBm"""
    # with np.errstate(divide='ignore'):
    return 10 * np.log10(watt * 1e3)

def np_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)

def standard_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)


# Coordinate system
def cart2cyl(pos: np.array):
    """ Transformation from cartesian to cylindrical coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), cartesian coordinate
    """
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    z = pos[:, 2]
    return np.vstack((rho, phi, z)).T


def cyl2cart(pos: np.array):
    """ Transformation from cylinder to cartesian coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), polar coordinate
    """
    pos = np.array(pos)
    x = pos[:, 0] * np.cos(pos[:, 1])
    y = pos[:, 0] * np.sin(pos[:, 1])
    z = pos[: , 2]
    return np.vstack((x, y, z)).T


def cart2spher(pos: np.array):
    """ Transformation from cartesian to cylindrical coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), cartesian coordinate
    """
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    theta = np.arccos(pos[:, 2] / rho)
    return np.vstack((rho, theta, phi)).T


def spher2cart(pos: np.array):
    """ Transformation from cylinder to cartesian coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), polar coordinate
    """
    x = pos[:, 0] * np.sin(pos[:, 1]) * np.cos(pos[:, 2])
    y = pos[:, 0] * np.sin(pos[:, 1]) * np.sin(pos[:, 2])
    z = pos[:, 0] * np.cos(pos[:, 1])
    return np.vstack((x, y, z)).T

class Position:
    # TODO: see if you can subclass numpy
    # todo: Add persistence
    """Wrapper for position vectors. Generally a K x 3 vector, where K is the number of points."""
    def __init__(self, cartesian: np.array = None, cylindrical: np.array = None, spherical: np.array = None, persistent: bool = False):
        """Constructor of the class.

        :param cartesian: np.array (K,3) representing the 3D Cartesian coordinates of K points.
        :param cylindrical: np.array (K,3) representing the 3D cylindrical coordinates of K points.
        :param spherical: np.array (K,3) representing the 3D spherical coordinates of K points.
        """
        if cartesian is not None:
            self._pos = cartesian
        elif cylindrical is not None:
            self._pos = cyl2cart(cylindrical)
        elif spherical is not None:
            self._pos = spher2cart(spherical)
        else:
            raise ValueError('No input has been provided.')

    @property
    def cart(self):
        """ Output cylindrical coordinates."""
        return self._pos

    @property
    def cyl(self):
        """ Output cylindrical coordinates."""
        rho = np.sqrt(self._pos[:, 0] ** 2 + self._pos[:, 1] ** 2)
        phi = np.arctan2(self._pos[:, 1], self._pos[:, 0])
        z = self._pos[:, 2]
        return np.vstack((rho, phi, z)).T

    @property
    def sph(self):
        """ Output spherical coordinates."""
        rho = np.sqrt(self._pos[:, 0] ** 2 + self._pos[:, 1] ** 2 + self._pos[:, 2] ** 2)
        phi = np.arctan2(self._pos[:, 1], self._pos[:, 0])
        theta = np.arccos(self._pos[:, 2] / rho)
        return np.vstack((rho, theta, phi)).T

    @property
    def norm(self):
        return np.linalg.norm(self._pos, axis=-1)

    @property
    def cartver(self):
        return (self.cart.T / self.norm).T    # transpose operation needed to obtain K x 3 vector

    @property
    def cylver(self):
        return (self.cyl.T / self.norm).T  # transpose operation needed to obtain K x 3 vector

    @property
    def sphver(self):
        return (self.sph.T / self.norm).T  # transpose operation needed to obtain K x 3 vector


    def __repr__(self):
        return self._pos.__repr__()

def euler_rotation_matrix(phi: float, theta: float, psi: float):
    return np.array([[np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi), -np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi), np.sin(theta)*np.sin(phi)],
                     [np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi), -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi), -np.sin(theta) * np.cos(phi)],
                     [np.sin(theta) * np.sin(psi), np.sin(theta) * np.cos(psi), np.cos(theta)]])


def gridmesh_2d(limits_x: tuple, limits_y: tuple, dx: float, dy: float, height = 0):
    num_points_x = int((limits_x[1] - limits_x[0]) / dx)
    num_points_y = int((limits_y[1] - limits_y[0]) / dy)
    num_points = num_points_x * num_points_y
    side_x_vec = np.arange(*limits_x, dx)
    side_y_vec = np.arange(*limits_y, dy)
    points = np.vstack((side_x_vec.repeat(num_points_x), np.tile(side_y_vec, num_points_y), height * np.ones(num_points))).T
    return points, num_points_x, num_points_y

# Print scenarios
def printplot(fig: plt.Figure = None,
              ax: plt.Axes or np.array = None,
              render: bool = False,
              filename: str = '',
              dirname: str = '',
              title: str or list = None,
              labels: list = None,
              grid: bool = True,
              orientation: str = 'vertical'):
    # Common print options with LaTeX type definitions
    rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    if fig is None:
        fig = plt.gcf()
        ax = plt.gca()
    if isinstance(ax, plt.Axes):    # Single axis plot
        if grid:
            ax.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_ylabel(labels[2])
        except (TypeError, IndexError):
            pass
        if ax.get_legend_handles_labels()[1]:   # Is there is at least a label, print the legend
            ax.legend()
        if not render:
            ax.set_title(title)
            fig.show()
            plt.close(fig=fig)
        else:
            filename = os.path.join(dirname, filename)
            ax.set_title(title)
            fig.savefig(filename + '.jpg', dpi=300)
            ax.set_title('')
            tikzplotlib.save(filename + '.tex', figure=fig)
            plt.close(fig=fig)
            return
    else:   # Multi-axes plot
        if ax[0].get_legend_handles_labels()[1]:  # Is there is at least a label, print the legend
            ax[0].legend()
        if grid:
            for a in ax:
                a.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            if orientation == 'vertical':
                ax[-1].set_xlabel(labels[0])
                for i, a in enumerate(ax):
                    a.set_ylabel(labels[i+1])
            else:   # Horizontal
                ax[0].set_ylabel(labels[1])
                for i, a in enumerate(ax):
                    a.set_xlabel(labels[0])
        except (TypeError, IndexError):
            pass
        # Keep the title for more than one figure
        try:
            for i, a in enumerate(ax):
                a.set_title(title[i])
        except IndexError:
            ax[0].set_title(title)
        if not render:
            fig.show()
            plt.close(fig=fig)
        else:
            filename = os.path.join(dirname, filename)
            fig.savefig(filename + '.jpg', dpi=300)
            tikzplotlib.save(filename + '.tex', figure=fig)


def std_progressbar(total_iteration, **kwargs):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True, **kwargs)

def standard_output_dir(subdirname: str) -> str:
    basedir = os.path.join(os.path.expanduser('~'), 'OneDrive/plots')
    if not os.path.exists(basedir):
        basedir = os.path.join(uppath(__file__, 2), 'plots')
        if not os.path.exists(basedir):
            os.mkdir(basedir)
    subdir = os.path.join(basedir, subdirname)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    output_dir = os.path.join(subdir, str(date.today()))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
