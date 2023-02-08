#!/usr/bin/env python3
# filename "nodes.py"

try:
    import cupy as np
except ImportError:
    import numpy as np
from scenario.common import Position

# standard values
UE_MAX_POW = 23.        # [dBm]
DIPOLE_GAIN = 2.15      # [dB]
BS_MAX_POW = 46.        # [dBm]
BS_ANT_GAIN = 12.85     # [dB]


class Node:
    """Construct a communication entity."""
    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 gain: float or np.ndarray = None,
                 max_pow: float or np.ndarray = None):
        """
        Parameters
        ---------
        :param n: int, number of nodes to place
        :param pos: ndarray of shape (n,3), position of the node in rectangular coordinates.
        :param gain : float, antenna gain of the node.
        :param max_pow : float, max power available on transmission in linear scale.
        """

        # Control on INPUT
        if pos.shape != (n, 3):
            raise ValueError(f'Illegal positioning: for Node, pos.shape must be ({n}, 3), instead it is {pos.shape}')

        # Set attributes
        self.n = n
        self.pos = Position(pos)
        self.gain = gain
        self.max_pow = max_pow


class BS(Node):
    """Base station class"""

    def __init__(self,
                 n: int = None,
                 pos: np.ndarray = None,
                 gain: float or np.ndarray = None,
                 max_pow: float or np.ndarray = None):
        """
        Parameters
        ---------
        :param n: number of BS.
        :param pos: ndarray of shape (n,3), position of the BS in rectangular coordinates.
        :param gain: float or ndarray, BS antenna gain. Default is 12.85 dB for all BS.
        :param max_pow : float or ndarray, BS max power. Default is 46 dBm for all UEs.
        """
        # Control on input
        if n is None:
            n = 1
        if gain is None:
            gain = BS_ANT_GAIN * np.ones(n)  # [dB]
        elif isinstance(gain, (float, int)):
            gain = gain * np.ones(n)
        else:
            assert gain.shape == (n,), "Gain array dimensions inconsistent"
        if max_pow is None:
            max_pow = BS_MAX_POW * np.ones(n)  # [dBm]
        elif isinstance(max_pow, (float, int)):
            max_pow = max_pow * np.ones(n)
        else:
            assert max_pow.shape == (n,), "Max power array dimensions inconsistent"

        # Init parent class
        super().__init__(n, pos, gain, max_pow)

    def __repr__(self):
        return f'BS-{self.n}'


class UE(Node):
    """User class
    """

    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 gain: float or np.ndarray = None,
                 max_pow: float or np.ndarray = None):
        """
        Parameters
        ---------
        :param n: number of UE.
        :param pos: ndarray of shape (n,3), position of the BS in rectangular coordinates.
        :param gain: float, BS antenna gain. Default is 2.15 dB.
        :param max_pow : float, BS max power. Default is 23 dBm.
        """
        # Control on input
        if gain is None:
            gain = DIPOLE_GAIN * np.ones(n)        # [dB]
        elif isinstance(gain, (float, int)):
            gain = gain * np.ones(n)
        else:
            assert gain.shape == (n,), "Gain array dimensions inconsistent"
        if max_pow is None:
            max_pow = UE_MAX_POW * np.ones(n)      # [dBm]
        elif isinstance(max_pow, (float, int)):
            max_pow = max_pow * np.ones(n)
        else:
            assert max_pow.shape == (n,), "Max power array dimensions inconsistent"

        # Init parent class
        super().__init__(n, pos, gain, max_pow)

    def __repr__(self):
        return f'UE-{self.n}'


class RIS(Node):
    """Reflective Intelligent Surface class"""

    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 num_els_h: int,
                 dist_els_h: float,
                 num_els_v: int = None,
                 dist_els_v: float = None,
                 orientation: str = None):
        """
        Parameters
        ---------
        :param n: number of RIS to consider # TODO: not all methods works for multi-RIS environment
        :param pos: ndarray of shape (n, 3), position of the RIS in rectangular coordinates.
        :param num_els_v: int, number of elements along z-axis.
        :param num_els_h: int, number of elements along x-axis.
        :param dist_els_v: float, size of each element.
        """
        # Default values
        if num_els_v is None:
            num_els_v = num_els_h
        if dist_els_v is None:
            dist_els_v = dist_els_h
        if orientation is None:
            orientation = 'xz'

        # Initialize the parent, having zero gain, max_pow is -np.inf,
        super().__init__(n, pos, 0.0, -np.inf)

        # Instance variables
        self.num_els = num_els_v * num_els_h  # total number of elements
        self.num_els_h = num_els_h  # horizontal number of elements
        self.num_els_v = num_els_v  # vertical number of elements
        self.dist_els_h = dist_els_h
        self.dist_els_v = dist_els_v

        # Compute RIS sizes
        self.size_h = num_els_h * self.dist_els_h  # horizontal size [m]
        self.size_v = num_els_v * self.dist_els_v  # vertical size [m]
        self.area = self.size_v * self.size_h   # area [m^2]

        # Element positioning
        self.m = np.tile(1 + np.arange(self.num_els_h), (self.num_els_v,))
        self.n = np.repeat(1 + np.arange(self.num_els_v), (self.num_els_h,))
        if orientation == 'xz':
            self.el_pos = np.vstack((self.dist_els_h * (self.m - (self.num_els_h + 1) / 2), np.zeros(self.num_els), self.dist_els_v * (self.n - (self.num_els_v + 1) / 2)))
        elif orientation == 'xy':
            self.el_pos = np.vstack((self.dist_els_h * (self.m - (self.num_els_h + 1) / 2), self.dist_els_v * (self.n - (self.num_els_v + 1) / 2), np.zeros(self.num_els)))
        elif orientation == 'yz':
            self.el_pos = np.vstack((np.zeros(self.num_els), self.dist_els_h * (self.m - (self.num_els_h + 1) / 2), self.dist_els_v * (self.n - (self.num_els_v + 1) / 2)))
        else:
            raise TypeError('Wrong orientation of the RIS')
        self.orientation = orientation

        # Configure RIS
        self.actual_conf = np.ones(self.num_els)    # initialized with attenuation and phase 0
        self.std_configs = None
        self.num_std_configs = None
        self.std_config_angles = None
        self.std_config_limits_plus = None
        self.std_config_limits_minus = None
        self.angular_resolution = None

    def ff_dist(self, wavelength):
        return 2 / wavelength * max(self.size_h, self.size_v) ** 2


    def init_std_configurations(self, wavelength: float, argument: float = None):
        """Set configurations offered by the RIS having a coverage of -3dB beamwidth on all direction from 0 to 180 degree

        :returns set_configs : ndarray, discrete set of configurations containing all possible angles (theta_s) in radians in which the RIS can steer the incoming signal.
        """
        if argument is None:
            argument = 1.391
        self.num_std_configs = int(np.ceil(self.num_els_h * self.dist_els_h * np.pi / wavelength / argument))
        self.std_configs = 1 - (2 * np.arange(1, self.num_std_configs + 1) - 1) * wavelength * argument / self.num_els_h / self.dist_els_h / np.pi
        if np.any(self.std_configs < - 1):
            self.std_configs = self.std_configs[:-1]
            self.num_std_configs -= 1
        self.std_config_limits_plus = 1 - (2 * np.arange(1, self.num_std_configs + 1)) * wavelength * argument / self.num_els_h / self.dist_els_h / np.pi
        self.std_config_limits_minus = 1 - (2 * np.arange(1, self.num_std_configs + 1) - 2) * wavelength * argument / self.num_els_h / self.dist_els_h / np.pi
        self.std_config_angles = np.arccos(self.std_configs)


    def set_std_configuration_2D(self, wavenumber, index, bs_pos: Position = Position(np.array([0, 10, 0]))):
        """Create the phase profile from codebook compensating the bs position and assuming attenuation 0"""
        # compensating bs
        phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
        phase_bs_v = np.cos(bs_pos.sph[:, 1])
        # Set standard configuration
        phase_conf_h = self.std_configs[index]
        phase_conf_v = 0
        # Compensating the residual phase
        phase_conf_tot = (self.num_els_h + 1) / 2 * self.dist_els_h * (phase_conf_h + phase_bs_h) + (self.num_els_v + 1) / 2 * self.dist_els_v * (phase_conf_v + phase_bs_v)
        # Put all together
        self.actual_conf = np.exp(1j * wavenumber * (phase_conf_tot - self.m * self.dist_els_h * (phase_conf_h + phase_bs_h) - self.n * self.dist_els_v * (phase_conf_v + phase_bs_v)))
        return self.actual_conf     #, phase_conf_h + phase_bs_h, phase_conf_v + phase_bs_v

    def set_configuration(self, wavenumber, configuration_angle, bs_pos: Position = Position(np.array([0, 10, 0]))):
        """Create the phase profile from codebook compensating the bs position and assuming attenuation 0"""
        # compensating bs
        phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
        phase_bs_v = np.cos(bs_pos.sph[:, 1])
        # Set specific configuration
        phase_c_h = np.cos(configuration_angle)
        phase_c_v = 0
        # Put all together
        self.actual_conf = np.exp(1j * wavenumber * (- self.m * self.dist_els_h * (phase_c_h + phase_bs_h) - self.n * self.dist_els_v * (phase_c_v + phase_bs_v)))
        return self.actual_conf

    def load_conf(self, wavenumber, azimuth_angle, elevation_angle, bs_pos: Position = Position(np.array([0, 10, 0]))):
        """Create the phase profile from codebook compensating the bs position and assuming attenuation 0"""
        if self.orientation == 'xy':    # configuration when the RIS is oriented xy
            # compensating bs
            phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
            phase_bs_v = np.sin(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
            # Set specific configuration
            phase_c_h = np.cos(azimuth_angle) * np.sin(elevation_angle)
            phase_c_v = np.sin(azimuth_angle) * np.sin(elevation_angle)
        elif self.orientation == 'xz':
            # compensating bs
            phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
            phase_bs_v = np.cos(bs_pos.sph[:, 1])
            # Set specific configuration
            phase_c_h = np.cos(azimuth_angle) * np.sin(elevation_angle)
            phase_c_v = np.cos(elevation_angle)
        else:
            raise ValueError('No configuration for the current RIS orientation')
        # Put all together
        self.actual_conf = np.exp(1j * wavenumber * (- self.m * self.dist_els_h * (phase_c_h + phase_bs_h) - self.n * self.dist_els_v * (phase_c_v + phase_bs_v)))
        return self.actual_conf

    def __repr__(self):
        return f'RIS-{self.n}'

    # TODO: print method for visualize the RIS


    # def set_angular_resolution(self):
    #     """Set RIS angular resolution. The observation space is ever considered to be 0 to pi/2 (half-plane) given our
    #     system setup.
    #
    #     Returns
    #     -------
    #     angular_resolution : float
    #         RIS angular resolution in radians given the number of configurations and uniform division of the observation
    #         space.
    #
    #     Example
    #     -------
    #     For num_configs = 4, angular_resolution evaluates to pi/8.
    #
    #     """
    #     self.angular_resolution = ((np.pi / 2) - 0) / self.num_configs

    # def indexing_els(self):
    #     """Define an array of tuples where each entry represents the ID of an element.
    #
    #     Returns
    #     -------
    #     id_els : ndarray of tuples of shape (self.num_els)
    #         Each ndarray entry has a tuple (id_v, id_h), which indexes the elements arranged in a planar array. Vertical
    #         index is given as id_v, while horizontal index is id_h.
    #
    #     Example
    #     -------
    #     For a num_els_v = 3 x num_els_h = 3 RIS, the elements are indexed as follows:
    #
    #                                             (2,0) -- (2,1) -- (2,2)
    #                                             (1,0) -- (1,1) -- (1,2)
    #                                             (0,0) -- (0,1) -- (0,2),
    #
    #     the corresponding id_els should contain:
    #
    #                     id_els = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)].
    #
    #     While the respective enumeration of the elements is:
    #
    #                                                 6 -- 7 -- 8
    #                                                 3 -- 4 -- 5
    #                                                 0 -- 1 -- 2,
    #
    #     the enumeration is stored at:
    #
    #                                         self.els_range = np.arange(num_els).
    #
    #     Therefore, id_els and self.els_range are two different index methods for the elements. The former is used to
    #     characterize the geometrical features of each element, while the latter is used for storage purposes.
    #     """
    #     # Get vertical ids
    #     id_v = self.els_range // self.num_els_v
    #
    #     # Get horizontal ids
    #     id_h = np.mod(self.els_range, self.num_els_h)
    #
    #     # Get array of tuples with complete id
    #     id_els = [(id_v[el], id_h[el]) for el in self.els_range]
    #
    #     return id_els
    #
    # def positioning_els(self):
    #     """Compute position of each element in the planar array.
    #
    #     Returns
    #     -------
    #
    #     """
    #     # Compute offsets
    #     offset_x = (self.num_els_h - 1) * self.size_el / 2
    #     offset_z = (self.num_els_v - 1) * self.size_el / 2
    #
    #     # Prepare to store the 3D position vector of each element
    #     pos_els = np.zeros((self.num_els, 3))
    #
    #     # Go through all elements
    #     for el in self.els_range:
    #         pos_els[el, 0] = (self.id_els[el][1] * self.size_el) - offset_x
    #         pos_els[el, 2] = (self.id_els[el][0] * self.size_el) - offset_z
    #
    #     return pos_els
    #
    # def plot(self):
    #     """Plot RIS along with the index of each element.
    #
    #     Returns
    #     -------
    #     None.
    #
    #     """
    #     fig, ax = plt.subplots()
    #
    #     # Go through all elements
    #     for el in self.els_range:
    #         ax.plot(self.pos_els[el, 0], self.pos_els[el, 2], 'x', color='black')
    #         ax.text(self.pos_els[el, 0] - 0.003, self.pos_els[el, 2] - 0.0075, str(self.id_els[el]))
    #
    #     # Plot origin
    #     ax.plot(0, 0, '.', color='black')
    #
    #     ax.set_xlim([np.min(self.pos_els[:, 0]) - 0.05, np.max(self.pos_els[:, 0]) + 0.05])
    #     ax.set_ylim([np.min(self.pos_els[:, 2]) - 0.05, np.max(self.pos_els[:, 2]) + 0.05])
    #
    #     ax.set_xlabel("x [m]")
    #     ax.set_ylabel("z [m]")
    #
    #     plt.show()

# class RxNoise:
#     """Represent the noise value at the physical receiver
#     # TODO: match with ambient noise and noise figure
#     """
#
#     def __init__(self, linear=None, dB=None, dBm: np.ndarray = np.array([-92.5])):
#         if (linear is None) and (dB is None):
#             self.dBm = dBm
#             self.dB = dBm - 30
#             self.linear = 10 ** (self.dB / 10)
#         elif (linear is not None) and (dB is None):
#             self.linear = linear
#             if self.linear != 0:
#                 self.dB = 10 * np.log10(self.linear)
#                 self.dBm = 10 * np.log10(self.linear * 1e3)
#             else:
#                 self.dB = -np.inf
#                 self.dBm = -np.inf
#         else:
#             self.dB = dB
#             self.dBm = dB + 30
#             self.linear = 10 ** (self.dB / 10)
#
#     def __repr__(self):
#         return (f'noise({self.linear:.3e}, '
#                 f'dB={self.dB:.1f}, dBm={self.dBm:.1f})')
