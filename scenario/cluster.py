#!/usr/bin/env python3
# filename "cluster.py"
import os.path as path

try:
    import cupy as np
except ImportError:
    import numpy as np
import scenario.common as cmn
from scenario.common import cluster_shapes, circular_uniform, semicircular_uniform, cyl2cart, cart2cyl, db2lin, channel_types
from scenario.nodes import UE, BS, RIS
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import rc
from scipy.constants import speed_of_light
from scipy.stats import rice, norm

# standard values
CARRIER = 3e9   # [Hz]
BW = 180e3      # [Hz]
N0B = -94       # [dBm]


class Cluster:
    """Creates a scenario defined by an area where UEs, BSs and RISs coexist.
    The coordinate system is centered in the center of the cluster.
    """

    def __init__(self,
                 shape: str,
                 sizes: float or np.ndarray,
                 carrier_frequency: float = CARRIER,
                 bandwidth: float = BW,
                 noise_power: float = N0B,
                 direct_channel: str = 'LoS',
                 reflective_channel: str = 'LoS',
                 pl_exponent: float = 2.,
                 d0: float = 1.,
                 rbs: int = 1,
                 symbols: int = 1,
                 rng: np.random.RandomState = None):

        """Constructor of class.

        Parameters
        __________
        :param sizes: float or np.ndarray, size of the cluster (radius if a circle, x,y,(z) side, if box, etc.)
        :param carrier_frequency: float, central frequency in Hertz.
        :param bandwidth: float, bandwidth in Hertz.
        :param rng: random nuber generator for reproducibility purpose
        """
        # Input check
        assert shape in cluster_shapes, f'Cluster kind not supported. Possible are: {cluster_shapes}'
        if isinstance(sizes, (float, int)):
            sizes = np.array([sizes, sizes, sizes])
        else:
            assert sizes.shape == (3,), f'Error in sizes definition: must be an array of shape (3,)'

        # Common print options with LaTeX type definitions
        rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)

        # Physical attributes
        self.shape = shape
        self.x_size = sizes[0]
        self.y_size = sizes[1]
        self.z_size = sizes[2]

        # Channel attributes
        self.direct_channel = direct_channel
        self.reflective_channel = reflective_channel
        self.pl_exponent = pl_exponent
        self.ref_dist = d0

        # Bandwidth available
        self.f0 = carrier_frequency
        self.wavelength = speed_of_light / carrier_frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.BW = bandwidth
        self.RBs = rbs
        self.freqs = self.f0 + self.BW * np.arange(self.RBs)

        # Random State generator
        self.rng = np.random.RandomState() if rng is None else rng

        # Noise Imposed to be -94 dBm
        self.N0B_dBm = noise_power
        self.N0B = float(cmn.dbm2watt(noise_power))

        # Nodes
        self.bs = None
        self.ue = None
        self.ris = None
        self.dist_br = None
        self.dist_ru = None
        self.dist_bu = None

        # Channels
        self.h_dir = None
        self.h_ris = None
        # TODO: transform the following into an attribute of ris class
        self.ris_array_factor = None

    @property
    def wavelengths(self) -> np.array:
        return speed_of_light / self.freqs

    def random_positioning(self, n: int):
        """Generate n positions in cartesian coordinate depending on the shape.

        :param n: int, number of points
        :return: np.array (n,3), position in cartesian coordinate of n points
        """
        if self.shape == 'circle':
            return cyl2cart(np.hstack((circular_uniform(n, self.x_size, rng=self.rng), np.zeros((n, 1)))))
        elif self.shape == 'semicircle':
            return cyl2cart(np.hstack((semicircular_uniform(n, self.x_size, rng=self.rng), np.zeros((n, 1)))))
        elif self.shape == 'box':
            x = self.x_size * self.rng.rand(n, 1) - self.x_size / 2
            y = self.y_size * self.rng.rand(n, 1) - self.y_size / 2
            return np.hstack((x, y, np.zeros((n, 1))))

    def check_position(self, pos: np.array):
        """Check if position is inside the cluster.

        :param pos: np.array (n,3), position to be tested
        :return: np.array (n), boolean indicating with True is the position is inside the cluster
        """
        if self.shape == 'circle':
            return cart2cyl(pos)[:, 0] <= self.x_size
        elif self.shape == 'semicircle':
            cyl_pos = cart2cyl(pos)
            return (cyl_pos[:, 0] <= self.x_size) & (cyl_pos[:, 1] <= np.pi) & (cyl_pos[:, 1] >= 0)
        elif self.shape == 'box':
            return (np.abs(pos[:, 0]) <= self.x_size/2) & (np.abs(pos[:, 1]) <= self.y_size/2)

    def place_bs(self,
                 n: int = 1,
                 position: np.array = None,
                 gain: float = None,
                 max_pow: float = None):
        """Place BS in the scenario. If a new set of BSs is set the old one is canceled.

        Parameters
        ----------
        :param n: number of BS to place,
        :param position: np.array shape (n,3), position of the bs in cartesian coordinate.
        :param gain: float, BS antenna gain G_b.
        :param max_pow: float, maximum power available at the BS.
        """
        # Input check
        if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number of nodes
            raise ValueError('n must be int >= 0.')
        if position is None:
            position = self.random_positioning(n)
        else:
            position = np.array(position)
            # if not np.any(self.check_position(position)):
            #     raise ValueError(f'Error in positioning for BS, recheck please.')

        # Append BS
        self.bs = BS(n=n, pos=position, gain=gain, max_pow=max_pow)

    def place_ue(self,
                 n: int,
                 position: np.array = None,
                 gain: float = None,
                 max_pow: float = None):
        """Place a predefined number n of UEs in the box. If a new set of UE is set the old one is canceled.

        Parameters
        ----------
        :param n: int, number of UEs to be placed.
        :param position: np.array shape (n,3), position of the UE in cartesian coordinate.
        :param gain: float, UE antenna gain G_k.
        :param max_pow: float, maximum power available at each UE.
        """
        # Input check
        if n <= 0:  # Cannot add a negative number of nodes
            raise ValueError('n must be int >= 0.')
        if position is None:
            position = self.random_positioning(n)
        else:
            position = np.array(position)
            if not np.any(self.check_position(position)):
                raise ValueError(f'Error in positioning for UEs, recheck please.')

        # Append UEs
        self.ue = UE(n=n, pos=position, gain=gain, max_pow=max_pow)

    def place_ris(self,
                  n: int,
                  position: np.ndarray = None,
                  num_els_x: int = None,
                  dist_els_x: float = None,
                  num_els_z: int = None,
                  dist_els_z: float = None,
                  orientation: str = None):
        """Place a set RIS in the scenario. If a new set is set the old one is canceled.

        Parameters
        ----------
        :param n: int, number of RIS to be placed.
        :param position : ndarray of shape (3,), position of the RIS in rectangular coordinates.
        :param num_els_x : int, number of elements along x-axis.
        :param dist_els_x: float, distance between elements on x-axis. Default: wavelength
        :param num_els_z: int, number of elements along z-axis. Default: as num_els_x
        :param dist_els_z: float, distance between elements on x-axis. Default: as dist_els_x
        """
        # Input check
        if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number of nodes
            raise ValueError('n must be int >= 0.')
        if position is None:
            position = np.array([0, 0, 0])
        else:
            position = np.array(position)
            if not np.any(self.check_position(position)):
                raise ValueError(f'Error in positioning for RIS, recheck please.')
        if num_els_x is None:
            num_els_x = 8
        if dist_els_x is None:
            dist_els_x = self.wavelength

        # Append RIS
        self.ris = RIS(n=n, pos=position, num_els_h=num_els_x, dist_els_h=dist_els_x, num_els_v=num_els_z, dist_els_v=dist_els_z, orientation=orientation)
        # Configure codebook RIS
        self.ris.init_std_configurations(self.wavelength)

    def compute_distances(self):
        self.dist_bu = np.linalg.norm(self.bs.pos.cart - self.ue.pos.cart, axis=1)[np.newaxis]
        self.dist_br = np.linalg.norm(self.bs.pos.cart - self.ris.pos.cart, axis=1)
        self.dist_ru = np.linalg.norm(self.ris.pos.cart - self.ue.pos.cart, axis=1)[np.newaxis]

    def get_channel_model(self):
        """Get Downlink (DL) and Uplink (UL) channel gain.

        Returns
        -------
        channel_gains_dl : ndarray of shape (num_configs, num_ues)
            Downlink channel gain between the BS and each UE for each RIS configuration.

        channel_gains_ul : ndarray of shape (num_configs, num_ues)
            Uplink channel gain between the BS and each UE for each RIS configuration.

        """
        # Compute DL pathloss component of shape (num_ues, )
        num = self.bs.gain * self.ue.gain * self.ris.dist_els_h ** 2
        den = (4 * np.pi * self.bs.distance * self.ue.distances)**2

        pathloss_dl = (num / den) * np.cos(self.bs.angles) ** 2

        # Compute UL pathloss component of shape (num_ues, )
        pathloss_ul = (num / den) * np.cos(self.ue.angles)**2

        # Compute constant phase component of shape (num_ues, )
        distances_sum = (self.bs.distance + self.ue.distances)
        disagreement = (np.sin(self.bs.angles) - np.sin(self.ue.angles)) * ((self.ris.num_els_h + 1) / 2) * self.ris.dist_els_h

        phi = - self.wavenumber * (distances_sum - disagreement)

        # Compute array factor of shape (num_configs, num_ues)
        enumeration_num_els_x = np.arange(1, self.ris.num_els_h + 1)
        sine_differences = (np.sin(self.ue.angles[np.newaxis, :, np.newaxis]) - np.sin(self.ris.configs[:, np.newaxis, np.newaxis]))

        argument = self.wavenumber * sine_differences * enumeration_num_els_x[np.newaxis, np.newaxis, :] * self.ris.dist_els_h

        array_factor_dl = self.ris.num_els_v * np.sum(np.exp(+1j * argument), axis=-1)
        array_factor_ul = array_factor_dl.conj()

        # Compute channel gains of shape (num_configs, num_ues)
        channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :]) * array_factor_dl
        channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :]) * array_factor_ul

        return channel_gains_dl, channel_gains_ul

    def get_channel_model_slotted_aloha(self):
        """Get Downlink (DL) and Uplink (UL) channel gain.

        Returns
        -------
        channel_gains_dl : ndarray of shape (1, num_ues)
            Downlink channel gain between the BS and each UE.

        channel_gains_ul : ndarray of shape (1, num_ues)
            Uplink channel gain between the BS and each UE.

        """
        # Compute DL pathloss component of shape (num_ues, )
        distance_bs_ue = np.linalg.norm(self.ue.pos - self.bs.pos, axis=-1)

        num = self.bs.gain * self.ue.gain
        den = (4 * np.pi * distance_bs_ue)**2

        pathloss_dl = num / den

        # Compute UL pathloss component of shape (num_ues, )
        pathloss_ul = pathloss_dl

        # Compute constant phase component of shape (num_ues, )
        phi = - self.wavenumber * distance_bs_ue

        # Compute channel gains of shape (num_configs, num_ues)
        channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :])
        channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :])

        return channel_gains_dl, channel_gains_ul

    def build_channels(self):
        """Build channels depending"""
        # TODO: save static values so you can compute only a part, reducing the time
        # TODO: control for input using try for the various cases
        # TODO: SIONNA.
        # Common values
        gain = self.bs.gain + self.ue.gain
        # Direct channel
        d_bu = np.linalg.norm(self.bs.pos.cart - self.ue.pos.cart, axis=1)[np.newaxis]
        # Path loss
        pl_bu = 20 * np.log10(4 * np.pi / self.wavelength) - gain + 10 * self.pl_exponent * np.log10(d_bu)
        # TODO: create a function with all the pl computations
        large_fad_bu = np.tile(db2lin(-pl_bu), (self.RBs, 1))
        small_fad_bu = fading(typ=self.direct_channel, dim=(self.RBs, self.ue.n))
        phase_shift_bu = 2 * np.pi / speed_of_light * self.freqs[np.newaxis].T @ d_bu
        self.h_dir = small_fad_bu * np.sqrt(large_fad_bu) * np.exp(- 1j * phase_shift_bu)

        # reflective channel
        d_br = np.linalg.norm(self.bs.pos.cart - self.ris.pos.cart, axis=1)
        d_ru = np.linalg.norm(self.ris.pos.cart - self.ue.pos.cart, axis=1)[np.newaxis]
        # Path loss
        pl_ru = 20 * np.log10(4 * np.pi) - gain - 20 * np.log10(self.ris.dist_els_h * self.ris.dist_els_v) + 20 * np.log10(d_br * d_ru)
        # TODO: what happens if RIS is not in the origins? There is the rotation matrix!
        # phases
        pos_factor = np.sin(self.ue.pos.sph[:, 2]) * np.sin(self.bs.pos.sph[:, 1])[np.newaxis]
        phase_shift_ru = self.freqs[np.newaxis].T * (d_ru - (self.ue.pos.cartver @ self.ris.el_pos).T)[np.newaxis].reshape((self.ris.num_els, 1, self.ue.n))
        phase_shift_br = self.freqs[np.newaxis].T * np.tile((d_br - self.bs.pos.cartver @ self.ris.el_pos)[np.newaxis].T, (1, self.RBs, self.ue.n))
        self.ris_array_factor = np.sum(self.ris.actual_conf[np.newaxis, np.newaxis].T * np.exp(- 1j * 2 * np.pi / speed_of_light * (phase_shift_ru + phase_shift_br)), axis=0)
        # window = np.tile(np.hamming(self.ris.num_els_h), self.ris.num_els_v)[np.newaxis, np.newaxis].T
        # self.ris_array_factor = np.sum(window * self.ris.actual_conf[np.newaxis, np.newaxis].T * np.exp(- 1j * 2 * np.pi / speed_of_light * (phase_shift_ru + phase_shift_br)), axis=0)
        # Overall
        small_fad_ru = fading(typ=self.reflective_channel, dim=(self.RBs, self.ue.n))
        large_fad_ru = np.tile(db2lin(-pl_ru), (self.RBs, 1))
        self.h_ris = self.ris_array_factor * pos_factor * small_fad_ru * np.sqrt(large_fad_ru)
        return self.h_dir + self.h_ris

    def compute_array_factor(self):
        """Compute the array factor based on the actual configuration"""
        # phases
        phase_shift_ru = self.freqs[np.newaxis].T * (self.dist_ru - (self.ue.pos.cartver @ self.ris.el_pos).T)[np.newaxis].reshape((self.ris.num_els, 1, self.ue.n))
        phase_shift_br = self.freqs[np.newaxis].T * np.tile((self.dist_br - self.bs.pos.cartver @ self.ris.el_pos)[np.newaxis].T, (1, self.RBs, self.ue.n))
        self.ris_array_factor = np.sum(self.ris.actual_conf[np.newaxis, np.newaxis].T * np.exp(- 1j * 2 * np.pi / speed_of_light * (phase_shift_ru + phase_shift_br)), axis=0)
        return self.ris_array_factor / self.ris.num_els


    def build_ris_channel(self, reset_array_factor: bool = False, return_small_large: bool = False):
        """Build RIS channel based on position and fading model"""
        # Path loss
        pl_bru = 10 * self.pl_exponent * np.log10(self.dist_br * self.dist_ru) - self.bs.gain - self.ue.gain - 40 * np.log10(self.wavelength / 4 / np.pi / self.ref_dist) - 20 * self.pl_exponent * np.log10(self.ref_dist)
        # phases
        element_factor = 1      # np.sin(self.ue.pos.sph[:, 2]) * np.sin(self.bs.pos.sph[:, 1])[np.newaxis]
        if reset_array_factor:
            self.compute_array_factor()
        # Overall
        small_fad_ru = fading(typ=self.reflective_channel, dim=(self.RBs, self.ue.n))
        large_fad_ru = np.tile(db2lin(-pl_bru), (self.RBs, 1))
        self.h_ris = self.ris_array_factor * element_factor * small_fad_ru * np.sqrt(large_fad_ru)
        if return_small_large:
            return self.h_ris / np.sqrt(large_fad_ru), np.squeeze(db2lin(-pl_bru))
        else:
            return self.h_ris

    def build_direct_channel(self, return_small_large: bool = False):
        """Build direct channel based on position and fading model"""
        # Path loss
        pl_bu = 20 * np.log10(4 * np.pi / self.wavelength) - self.bs.gain + self.ue.gain + 10 * self.pl_exponent * np.log10(self.dist_bu)
        large_fad_bu = np.tile(db2lin(-pl_bu), (self.RBs, 1))
        small_fad_bu = fading(typ=self.direct_channel, dim=(self.RBs, self.ue.n))
        phase_shift_bu = 2 * np.pi / speed_of_light * self.freqs[np.newaxis].T @ self.dist_bu
        self.h_dir = small_fad_bu * np.sqrt(large_fad_bu) * np.exp(- 1j * phase_shift_bu)
        if return_small_large:
            return self.h_dir / np.sqrt(large_fad_bu) , np.squeeze(db2lin(-pl_bu))
        else:
            return  self.h_dir

    def build_ris_ch_tensor(self, reset_array_factor: bool = False):
        """Build RIS channel considering the whole tensor dimensions C x F x K
        Not implemented yet
        """
        # TODO: try to separate as much as possible the computation of the various parts
        self.build_ris_channel(reset_array_factor)
        return self.h_ris

    def plot_scenario(self, render: bool = False, **kwargs):
        """This method will plot the scenario of communication"""
        # kwargs input
        if ~render:
            file = None
            folder = None
        else:
            try:
                file = kwargs['filename']
            except KeyError:
                file = 'scenario'
            try:
                folder = kwargs['dirname']
            except KeyError:
                folder = path.join(path.dirname(__file__), 'data')
        # Open axes
        fig, ax = plt.subplots()

        # Displacement value
        delta = self.x_size / 100
        # Box positioning
        if self.shape == 'box':
            shape = plt.Rectangle((-self.x_size / 2, -self.y_size / 2), self.x_size, self.y_size, ec="black", ls="--", lw=1, fc='#45EF0605')
            ax.set_ylim(- self.y_size - delta, self.y_size + delta)
            ax.set_xlim(- self.x_size - delta, self.x_size + delta)
            ax.axis('equal')
        elif self.shape == 'circle':
            shape = plt.Circle((0, 0), self.x_size, ec="black", ls="--", lw=1, fc='#45EF0605')
            ax.set_ylim(- self.x_size - delta, self.x_size + delta)
            ax.set_xlim(- self.x_size - delta, self.x_size + delta)
            ax.axis('equal')
        else: # or self.shape == 'semicircle':
            shape = plt.Circle((0, 0), self.x_size, ec="black", ls="--", lw=1, fc='#45EF0605')
            ax.set_ylim(- delta, self.x_size + delta)
            ax.set_xlim(- self.x_size - delta, self.x_size + delta)
            ax.axis('scaled')
        ax.add_patch(shape)
        # User positioning
        try:
            # BS        
            ax.scatter(self.bs.pos.cart[:, 0], self.bs.pos.cart[:, 1], c=cmn.node_color['BS'], marker=cmn.node_mark['BS'], label='BS')
            # plt.text(self.bs.pos[:, 0], self.bs.pos[:, 1] + delta, s='BS', fontsize=10)
            # UE
            ax.scatter(self.ue.pos.cart[:, 0], self.ue.pos.cart[:, 1], c=cmn.node_color['UE'], marker=cmn.node_mark['UE'], label='UE')
            for k in np.arange(self.ue.n):
                ax.text(self.ue.pos.cart[k, 0], self.ue.pos.cart[k, 1] + delta, s=f'{k}', fontsize=10)
            # RIS
            ax.scatter(self.ris.pos.cart[:, 0], self.ris.pos.cart[:, 1], c=cmn.node_color['RIS'], marker=cmn.node_mark['RIS'], label='RIS')
            # plt.text(self.ris.pos[:, 0], self.ris.pos[:, 1] + delta, s='RIS', fontsize=10)
        except TypeError:
            # BS        
            ax.scatter(self.bs.pos.cart[:, 0].get(), self.bs.pos.cart[:, 1].get(), c=cmn.node_color['BS'], marker=cmn.node_mark['BS'], label='BS')
            # plt.text(self.bs.pos[:, 0], self.bs.pos[:, 1] + delta, s='BS', fontsize=10)
            # UE
            ax.scatter(self.ue.pos.cart[:, 0].get(), self.ue.pos.cart[:, 1].get(), c=cmn.node_color['UE'], marker=cmn.node_mark['UE'], label='UE')
            for k in np.arange(self.ue.n):
                ax.text(self.ue.pos.cart[k, 0].get(), self.ue.pos.cart[k, 1].get() + delta.get(), s=f'{k}', fontsize=10)
            # RIS
            ax.scatter(self.ris.pos.cart[:, 0].get(), self.ris.pos.cart[:, 1].get(), c=cmn.node_color['RIS'], marker=cmn.node_mark['RIS'], label='RIS')
            # plt.text(self.ris.pos[:, 0], self.ris.pos[:, 1] + delta, s='RIS', fontsize=10)
        
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Finally
        cmn.printplot(fig, ax, render, filename=file, dirname=folder, labels=['$x$ [m]', '$y$ [m]'])




# Fading channel
def fading(typ: str, dim: tuple = (1,),
           shape: float = 6, seed: int = None) -> np.ndarray:
    """Create a sampled fading channel from a given distribution and given
    dimension.

    Parameters
    __________
    typ : str in dic.channel_types,
        type of the fading channel to be used.
    dim : tuple,
        dimension of the resulting array.
    shape : float [dB],
        shape parameters used in the rice distribution modeling the power of
        the LOS respect to the NLOS rays.
    seed : int,
        seed used in the random number generator to provide the same arrays if
        the same is used.
    """

    if typ not in channel_types:
        raise ValueError(f'Type can only be in {channel_types}')
    elif typ == 'AWGN' or typ == 'LoS':
        return np.ones(dim)
    elif typ == 'No':
        return np.zeros(dim)
    elif typ == "Rayleigh":
        vec = norm.rvs(size=2 * np.prod(dim), random_state=seed)
        return (vec[0:1] + 1j * vec[1:2]).reshape(dim) / np.sqrt(2)  # TODO FIXING THIS BUG
    elif typ == "Rice":
        return rice.rvs(10 ** (shape / 10), size=np.prod(dim), random_state=seed).reshape(dim)  # TODO: FIX ALSO THIS
    elif typ == "Shadowing":
        return norm.rvs(scale=10 ** (shape / 10), random_state=seed)