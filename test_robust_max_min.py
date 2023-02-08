import os.path as path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.stats import ncx2

import scenario.common as cmn
from environment import RISFrequencyEnv2D, command_parser, NUM_EL_X, CARRIER_FREQ


def min_max_heu(snr, rho):
    # Check input
    assert snr.shape == rho.shape, "SNR dimensions differ for RHO ones"
    # Retrieve data
    num_resources = snr.shape[0]
    num_users = snr.shape[1]
    # Special cases
    if num_users == 0:
        return [], []
    elif num_users == 1:
        rho[:] = True
        rk = cp.sum(cp.log2(1 + rho * snr), axis=0)
        return rk, rho
    # Algorithm initialization
    A = cp.ones(num_resources)
    # Assign at least one resource to each user
    for k in np.arange(num_users):
        n = cp.nanargmax(A * snr[:, k])
        rho[n, k] = True
        A[n] = cp.nan
    # Compute resulting rate
    rk = cp.sum(cp.log2(1 + rho * snr), axis=0)
    # Distribute the remaining resources
    while not cp.all(cp.isnan(A)):
        k = rk.argmin()
        n = cp.nanargmax(A * snr[:, k])
        rho[n, k] = True
        A[n] = cp.nan
        rk[k] += cp.log2(1 + snr[n, k])
    return rk, rho


# Cell
rad = 30.   # radius of the cell
rad_min = np.ceil(NUM_EL_X ** 2  / 2 * c / CARRIER_FREQ)  # ff distance
bs_pos = np.array([[10, 100, 0]])
# Generate a dummy environment to get the number of std configuration
env = RISFrequencyEnv2D(5, bs_pos, np.array([[0,0,0]]))
Cmax = env.ris.num_std_configs

# Transmission power
p_dBm = 0  # [dBm]
power = cmn.dbm2watt(p_dBm)

# Blockage factor on direct channel
kappa_dB_vec = np.arange(-12., 15, 3)      # dB
overloading_factor = 1

# Probability of successful transmission
epsilon = 0.95

# Users under test
ue_init = 50
quant = 100

# Scenario for averaging results
batch_size = 50
save_every = batch_size / 10

if __name__ == '__main__':
    # input parser
    render, datadir = command_parser()

    resources = 50
    prefix = f'RBs{resources:03d}_robust_max_min_'
    users = np.arange(ue_init, overloading_factor * resources * Cmax + quant, quant)

    for kappa_dB in kappa_dB_vec:
        print(f'Kappa factor = {kappa_dB}')
        # Compute linear kappa
        kappa = cmn.db2lin(kappa_dB)

        if render:
            # Load previous data if they exist
            filename = path.join(datadir, f'{prefix}kappa{kappa_dB:02.0f}.npz')
        else:
            filename = ''
        # Pre-allocating the parameters to be shown
        try:
            data = np.load(filename)
            start = int(data['last'] + 1)
            avg_throughput_jnt = data['avg_throughput_jnt']
            avg_throughput_seq = data['avg_throughput_seq']
            avg_throughput_bnc = data['avg_throughput_bnc']
            fairness_jnt = data['fairness_jnt']
            fairness_seq = data['fairness_seq']
            fairness_bnc = data['fairness_bnc']
            ues_allocated_jnt = data['ues_allocated_jnt']
            ues_allocated_seq = data['ues_allocated_seq']
            ues_allocated_bnc = data['ues_allocated_bnc']
        except FileNotFoundError:
            start = 0
            avg_throughput_jnt = np.zeros((batch_size, len(users)))
            avg_throughput_seq = np.zeros((batch_size, len(users)))
            avg_throughput_bnc = np.zeros((batch_size, len(users)))
            fairness_jnt = np.zeros((batch_size, len(users)))
            fairness_seq = np.zeros((batch_size, len(users)))
            fairness_bnc = np.zeros((batch_size, len(users)))
            ues_allocated_jnt = np.zeros((batch_size, len(users)))
            ues_allocated_seq = np.zeros((batch_size, len(users)))
            ues_allocated_bnc = np.zeros((batch_size, len(users)))


        # Varying experiments
        for k_batch in cmn.std_progressbar(range(start, batch_size), desc='batch num'):
            #  Set the generator
            rng = cp.random.RandomState(k_batch)

            # Varying UEs
            for k_u, u in enumerate(users):
                # Generate user positioning
                d = cp.sqrt((rad ** 2 - rad_min ** 2) * rng.rand(u, 1) + rad_min ** 2)
                phi = cp.pi * rng.rand(u, 1)
                ue_pos = cp.hstack((d * cp.cos(phi), d * cp.sin(phi), cp.zeros((u, 1))))

                # Build environment
                env = RISFrequencyEnv2D(rad, bs_pos, ue_pos, rbs=resources, rng=rng)

                # Optimization based on position
                best_configurations = env.best_configuration_selection()

                # Compute LoS channels
                h_dir, omega = env.build_direct_channel(return_small_large=True)
                h_dir = cp.repeat(h_dir[cp.newaxis], Cmax, axis=0)

                h_ris = cp.zeros((Cmax, env.RBs, u), dtype=np.complex128)
                for i in range(Cmax):
                    env.set_std_configuration(i)
                    h_ris[i] = env.build_ris_channel(reset_array_factor=True)

                # Insert NLoS channels
                eta = cp.sqrt(omega * kappa / (kappa + 1)) * h_dir + h_ris
                sigma = cp.sqrt(omega / (kappa + 1))
                ran = cp.asarray(rng.randn(*h_ris.shape) + 1j * rng.randn(*h_ris.shape)) / cp.sqrt(2)
                h_all = eta + sigma * ran

                # Compute SNR
                gamma_all = np.abs(h_all) ** 2 * power / env.N0B

                # Chi-squared robust evaluation
                nc = 2 * cp.asnumpy(cp.abs(eta / sigma) ** 2)
                gamma_eps = power / env.N0B * cp.asarray(ncx2.ppf(1 - epsilon, 2, nc)) * sigma ** 2 / 2

                ### Allocation problem
                # Reset allocation variable
                rho_jnt = cp.zeros((Cmax, env.RBs, u), dtype=bool)      # joint allocation variable
                rho_seq = cp.zeros((Cmax, env.RBs, u), dtype=bool)      # sequential allocation variable
                rho_bnc = cp.zeros((Cmax, env.RBs, u), dtype=bool)      # benchmark allocation variable

                # Reshape
                gamma_eps = cp.reshape(gamma_eps, (Cmax * env.RBs, u))
                rho_jnt = cp.reshape(rho_jnt, (Cmax * env.RBs, u))
                gamma_all = cp.reshape(gamma_all, (Cmax * env.RBs, u))
                rho_bnc = cp.reshape(rho_bnc, (Cmax * env.RBs, u))

                # Joint and benchmark allocation strategies
                rk_jnt, rho_jnt = min_max_heu(gamma_eps, rho_jnt)
                rk_bnc, rho_bnc = min_max_heu(gamma_all, rho_bnc)

                # Revert back
                gamma_eps = cp.reshape(gamma_eps, (Cmax, env.RBs, u))
                rho_jnt = cp.reshape(rho_jnt, (Cmax, env.RBs, u))
                gamma_all = cp.reshape(gamma_eps, (Cmax, env.RBs, u))
                rho_bnc = cp.reshape(rho_bnc, (Cmax, env.RBs, u))

                # Sequential allocation
                rk_seq = cp.zeros((Cmax, u))
                # RIS sep allocation
                for i in range(Cmax):
                    # find users in conf i
                    allocation = best_configurations == i
                    if cp.sum(allocation) == 0:
                        continue
                    elif cp.sum(allocation) > env.RBs:
                        ue_split = min(cp.sum(allocation) // 2, env.RBs)
                        num_split = int(cp.sum(allocation) - ue_split) // env.RBs
                        # Allocate the first RBs users in the current time slot
                        first_allocation = np.argwhere(allocation == True)[:ue_split, 0]
                        rk_seq[i][first_allocation], rho_seq[i][:, first_allocation] = min_max_heu(
                        gamma_eps[i][:, first_allocation], rho_seq[i][:, first_allocation])
                        for _ in range(num_split + 1):
                            # Allocate the remaining users in a new time slot
                            rk_mid = cp.zeros_like(rk_seq[i])
                            rho_mid = cp.zeros_like(rho_seq[i])
                            mid_allocation = np.argwhere(allocation == True)[ue_split:, 0]
                            rk_mid[mid_allocation], rho_mid[:, mid_allocation] = min_max_heu(gamma_eps[i][:, mid_allocation], cp.zeros_like(rho_seq)[i][:, mid_allocation])
                            # Add the new time slot in the end
                            rk_seq = cp.vstack((rk_seq, rk_mid[cp.newaxis]))
                            rho_seq = cp.vstack((rho_seq, rho_mid[cp.newaxis]))
                        del rk_mid, rho_mid
                    else:
                        rk_seq[i, allocation], rho_seq[i][:, allocation] = min_max_heu(gamma_eps[i][:, allocation], rho_seq[i][:, allocation])
                rk_seq = cp.sum(rk_seq, axis=0)

                # Count the actual configuration used
                Cused_jnt = cp.count_nonzero(cp.sum(rho_jnt, axis=[1, 2]))
                Cused_seq = cp.count_nonzero(cp.sum(rho_seq, axis=[1, 2]))
                Cused_bnc = cp.count_nonzero(cp.sum(rho_bnc, axis=[1, 2]))

                # Compute rates
                rk_jnt *= env.BW / Cused_jnt
                rk_seq *= env.BW / Cused_seq
                rk_bnc *= env.BW / Cused_bnc

                fairness_jnt[k_batch, k_u] = cp.asnumpy(cp.sum(rk_jnt) ** 2 / cp.sum(rk_jnt ** 2) / u)
                fairness_seq[k_batch, k_u] = cp.asnumpy(cp.sum(rk_seq) ** 2 / cp.sum(rk_seq ** 2) / u)
                fairness_bnc[k_batch, k_u] = cp.asnumpy(cp.sum(rk_bnc) ** 2 / cp.sum(rk_bnc ** 2) / u)

                avg_throughput_jnt[k_batch, k_u] = cp.asnumpy(cp.sum(rk_jnt))
                avg_throughput_seq[k_batch, k_u] = cp.asnumpy(cp.sum(rk_seq))
                avg_throughput_bnc[k_batch, k_u] = cp.asnumpy(cp.sum(rk_bnc))

                # Count the allocated users
                ues_allocated_jnt[k_batch, k_u] = cp.count_nonzero(cp.sum(rho_jnt, axis=[0, 1]))
                ues_allocated_seq[k_batch, k_u] = cp.count_nonzero(cp.sum(rho_seq, axis=[0, 1]))
                ues_allocated_bnc[k_batch, k_u] = cp.count_nonzero(cp.sum(rho_bnc, axis=[0, 1]))

            if render:  # SAVE DATA
                if np.mod(k_batch, save_every) == 0 or k_batch == batch_size - 1:
                    np.savez(filename, last=k_batch, fairness_jnt=fairness_jnt, fairness_seq=fairness_seq, fairness_bnc=fairness_bnc,
                             avg_throughput_jnt=avg_throughput_jnt, avg_throughput_seq=avg_throughput_seq, avg_throughput_bnc=avg_throughput_bnc,
                             ues_allocated_jnt=ues_allocated_jnt, ues_allocated_seq=ues_allocated_seq, ues_allocated_bnc=ues_allocated_bnc)


        if not render:
            # Take the average over the batch size
            avg_throughput_jnt = np.mean(avg_throughput_jnt, axis=0)
            avg_throughput_seq = np.mean(avg_throughput_seq, axis=0)
            avg_throughput_bnc = np.mean(avg_throughput_bnc, axis=0)
            fairness_jnt = np.mean(fairness_jnt, axis=0)
            fairness_seq = np.mean(fairness_seq, axis=0)
            fairness_bnc = np.mean(fairness_bnc, axis=0)

            # Figure plot
            plt.plot(users, avg_throughput_jnt / 1e6, label='RIS (jointly)')
            plt.plot(users, avg_throughput_seq / 1e6, label='RIS (sep)')
            plt.plot(users, avg_throughput_bnc / 1e6, label='RIS CSI')
            cmn.printplot(labels=['$K$', 'throughput [Mbit/s]'])

            plt.plot(users, fairness_jnt, label='RIS (jointly)')
            plt.plot(users, fairness_seq, label='RIS (sep)')
            plt.plot(users, fairness_bnc, label='RIS CSI')
            cmn.printplot(labels=['$K$', 'Fairness'])
