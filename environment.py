import os.path

from scenario.cluster import Cluster
import scenario.common as cmn
try:
    import cupy as np
except ImportError:
    import numpy as np
from scipy.constants import c
import argparse


# GLOBAL STANDARD PARAMETERS
OUTPUT_DIR = cmn.standard_output_dir('ris-frequency')
DATADIR = os.path.join(os.path.dirname(__file__), 'data')
# Set parameters
NUM_EL_X = 10
CARRIER_FREQ = 1.8e9        # [Hz]
BANDWIDTH = 180e3           # [Hz]
PRBS = 130
NOISE_FIGURE = 9            # [dB]
PL_EXPONENT = 2.7

# Parser
def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", action="store_true", default=False)
    parser.add_argument("-d", "--directory", default=DATADIR)
    args: dict = vars(parser.parse_args())
    return list(args.values())


# Class
class RISFrequencyEnv2D(Cluster):
    def __init__(self,
                 radius: float,
                 bs_position: np.array,
                 ue_position: np.array or int,
                 ris_num_els: int = NUM_EL_X,
                 carrier_frequency: float = CARRIER_FREQ,
                 bandwidth: float = BANDWIDTH,
                 rbs: int = PRBS,
                 noise_figure: float = NOISE_FIGURE,
                 pl_exponent: float = PL_EXPONENT,
                 time_slots: int = None,
                 rng: np.random.RandomState = None):
        # Compute noise power from RB
        noise_power = float(cmn.watt2dbm(cmn.thermal_noise(rbs * bandwidth, noise_figure=noise_figure)))
        # Init parent class
        super().__init__(shape='semicircle',
                         sizes=radius,
                         carrier_frequency=carrier_frequency,
                         bandwidth=bandwidth,
                         noise_power=noise_power,
                         direct_channel='LoS',
                         reflective_channel='LoS',
                         pl_exponent=pl_exponent,
                         rbs=rbs,
                         rng=rng)
        self._int_tested = 25       # attribute related to the number of integers under frequency_scheduling
        self.place_bs(1, bs_position)
        self.place_ue(ue_position.shape[0], ue_position)
        self.place_ris(1, np.array([[0, 0, 0]]), num_els_x=ris_num_els, dist_els_x=self.wavelength/2)
        self.compute_distances()


    def set_std_configuration(self, index: int):
        return self.ris.set_std_configuration_2D(self.wavenumber, index, self.bs.pos)

    def load_conf(self, angle: float):
        return self.ris.load_conf(self.wavenumber, azimuth_angle=angle, elevation_angle=np.pi / 2, bs_pos=self.bs.pos)

    def best_configuration_selection(self):
        """Find the best configuration considering the azimuth angle only"""
        return np.abs(self.ris.std_config_angles[np.newaxis].T - self.ue.pos.sph[:, 2]).argmin(axis=0)

    def best_frequency_selection(self):
        """Find the best frequency for the transmission (residual phase shift on f0 MUST BE NULL)"""
        dist_triangle = self.ue.pos.norm + self.bs.pos.norm - np.linalg.norm(self.ue.pos.cart - self.bs.pos.cart[0], axis=1)
        integer_multiplier = np.ceil(self.f0 / c * dist_triangle) + np.arange(self._int_tested)[np.newaxis].T
        return np.round((c * integer_multiplier / dist_triangle - self.f0) / self.BW).T.astype(int)
        # Old version
        # ru = env.ue.pos.norm[u]
        # rub = np.linalg.norm(env.ue.pos.cart[u] - env.bs.pos.cart[0])
        # triangle = ru + rb - rub
        # # for i in range(env.ris.num_std_configs)[:-1]:
        # i = best_configurations[u]
        # Set configuration
        # _, varphi_x, varphi_z = env.set_std_configuration(best_configurations[u])
        # compute the right periodicity of the phase vs frequency
        # phase_sum = 0  # - (env.ris.num_els_h + 1) / 2 * env.ris.dist_els_h * varphi_x - (env.ris.num_els_v + 1) / 2 * env.ris.dist_els_v * varphi_z
        # k_min = np.floor(env.f0 / c * (phase_sum - triangle))
        # k = np.arange(k_min, k_min - 400, -1)
        # f = np.round(((env.f0 * phase_sum - c * k) / triangle - env.f0) / env.BW)

    def coherence_RB_indexes(self):
        """Coherence bandwidth (residual phase shift on f0 MUST BE NULL)

        :return tuple(np.array, np.array): containing the lower bound and upper bound RB indexes that guarantees np.abs(h_tot/h_dir) ** 2 > 1 (now it is even more but let's see in the future...
        # TODO:
            - check if it is better to implement the > 0 coherence
            - implement warning to check if the np.min(coherence_tuple[0][:, -1]) < env.RBs to increase the int_tested value
            - implement a vectorized form of the boolean index matrix
        """
        # Compute the
        dist_triangle = self.dist_ru + self.dist_br - self.dist_bu
        integer_multiplier_lo = np.ceil(self.f0 / c * dist_triangle + 1/6) + np.arange(self._int_tested)[np.newaxis].T
        integer_multiplier_up = np.ceil(self.f0 / c * dist_triangle - 1 / 6) + np.arange(self._int_tested)[np.newaxis].T
        RB_index_lo = np.round((c * (6 * integer_multiplier_lo - 1) / dist_triangle / 6 - self.f0) / self.BW).T.astype(int)
        RB_index_up = np.round((c * (6 * integer_multiplier_up + 1) / dist_triangle / 6 - self.f0) / self.BW).T.astype(int)

        # Fix the lower bound:
        # if RB_index_up[:, 0] < RB_index_lo[:, 0] it means that the cosine starts in the positive part. For these cases, the RB index = 0 is the actual first lower bound.
        index = RB_index_up[:, 0] < RB_index_lo[:, 0]
        RB_index_lo[index] = np.roll(RB_index_lo[index], 1, axis=1)
        RB_index_lo[index, 0] = 0

        # # Build boolean coherence RB index matrix (slow version)
        # RB_indexes = np.arange(self.RBs)
        # coherence_RB_var = np.zeros((self.RBs, self.ue.n), dtype=bool)
        # for u in range(self.ue.n):
        #     num_tests = sum(RB_index_lo[u] < self.RBs)
        #     for i in range(num_tests):
        #         coherence_RB_var[:, u] += (RB_index_lo[u, i] <= RB_indexes) & (RB_indexes <= RB_index_up[u, i])
        # Build boolean coherence RB index matrix (faster version)
        # RB_indexes = np.arange(self.RBs)
        # coherence_RB_var = np.zeros((self.RBs, self.ue.n), dtype=bool)
        # for u in range(self.ue.n):
        #     coherence_RB_var[:, u] = np.sum(((np.tile(RB_index_lo[u][np.newaxis], (self.RBs, 1)).T <= RB_indexes) & (RB_indexes <= np.tile(RB_index_up[u][np.newaxis], (self.RBs, 1)).T)), axis=0)

        # Build boolean coherence RB index matrix (vectorized version)
        RB_indexes = np.arange(self.RBs)
        coherence_RB_var = np.sum((np.tile(RB_index_lo[np.newaxis].reshape(self.ue.n, self._int_tested, 1), (1, 1, self.RBs)) <= RB_indexes) &
                                   (RB_indexes <= np.tile(RB_index_up[np.newaxis].reshape(self.ue.n, self._int_tested, 1), (1, 1, self.RBs))), axis=1).squeeze().astype(bool).T
        return  RB_index_lo, RB_index_up, coherence_RB_var
