import os.path as path

import matplotlib.pyplot as plt
import numpy as np

import scenario.common as cmn
from environment import command_parser, OUTPUT_DIR, RISFrequencyEnv2D
from test_robust_max_rate import kappa_dB_vec, epsilon


bs_pos = np.array([[10, 100, 0]])
# Generate a dummy environment to get the number of std configuration
env = RISFrequencyEnv2D(5, bs_pos, np.array([[0,0,0]]))
Cmax = env.ris.num_std_configs
resources = 50

if __name__ == '__main__':
    # input parser
    render,  datadir = command_parser()

    names = ['max_rate', 'max_min']



    for n in names:

        rate_vs_kappa_jnt = np.zeros(len(kappa_dB_vec))
        rate_vs_kappa_seq = np.zeros(len(kappa_dB_vec))
        rate_vs_kappa_bnc = np.zeros(len(kappa_dB_vec))
        fair_vs_kappa_jnt = np.zeros(len(kappa_dB_vec))
        fair_vs_kappa_seq = np.zeros(len(kappa_dB_vec))
        fair_vs_kappa_bnc = np.zeros(len(kappa_dB_vec))


        for i, kappa_dB in enumerate(kappa_dB_vec):
            print(f'Rice factor = {kappa_dB}')
            # Load data
            filename = path.join(datadir, f'RBs{resources:03d}_robust_{n}_kappa{kappa_dB:02.0f}.npz')

            # Load data
            data = np.load(filename)
            avg_throughput_jnt = data['avg_throughput_jnt']
            avg_throughput_seq = data['avg_throughput_seq']
            avg_throughput_bnc = data['avg_throughput_bnc']
            fairness_jnt = data['fairness_jnt']
            fairness_seq = data['fairness_seq']
            fairness_bnc = data['fairness_bnc']
            ues_allocated_jnt = data['ues_allocated_jnt']
            ues_allocated_seq = data['ues_allocated_seq']
            ues_allocated_bnc = data['ues_allocated_bnc']

            if n == 'max_rate':
                from test_robust_max_rate import overloading_factor
                users = np.array(resources * Cmax * overloading_factor, dtype=int)
            else:
                from test_matching_max_min import ue_init, quant, overloading_factor
                users = np.arange(ue_init, overloading_factor * resources * Cmax + quant, quant)

            # compute some data
            tau_csi = np.ceil(users * (env.ris.num_els_h + 1) / 2)
            N_ofdm_csi = tau_csi
            N_ofdm = 14
            ratio_csi = Cmax * N_ofdm / (Cmax * N_ofdm + N_ofdm_csi)
            loc_over_dat = 7/N_ofdm
            ratio_loc = epsilon * 1 / (1 + loc_over_dat)

            # Compute the average rate per user
            throughput_per_ue_jnt = np.mean(avg_throughput_jnt / ues_allocated_jnt, axis=0) / 1e6 * ratio_loc
            throughput_per_ue_seq = np.mean(avg_throughput_seq / ues_allocated_seq, axis=0) / 1e6 * ratio_loc
            throughput_per_ue_bnc = np.mean(avg_throughput_bnc / ues_allocated_bnc, axis=0) / 1e6 * ratio_csi

            # Take the average over the batch size
            avg_throughput_jnt = np.mean(avg_throughput_jnt, axis=0) / 1e6 * ratio_loc
            avg_throughput_seq = np.mean(avg_throughput_seq, axis=0) / 1e6 * ratio_loc
            avg_throughput_bnc = np.mean(avg_throughput_bnc, axis=0) / 1e6 * ratio_csi
            fairness_jnt = np.mean(fairness_jnt, axis=0)
            fairness_seq = np.mean(fairness_seq, axis=0)
            fairness_bnc = np.mean(fairness_bnc, axis=0)

            k = 0
            rate_vs_kappa_jnt[i] = avg_throughput_jnt[k]
            rate_vs_kappa_seq[i] = avg_throughput_seq[k]
            rate_vs_kappa_bnc[i] = avg_throughput_bnc[k]
            fair_vs_kappa_jnt[i] = fairness_jnt[k]
            fair_vs_kappa_seq[i] = fairness_seq[k]
            fair_vs_kappa_bnc[i] = fairness_bnc[k]

            rate_vs_kappa_jnt[i] = throughput_per_ue_jnt[k]
            rate_vs_kappa_seq[i] = throughput_per_ue_seq[k]
            rate_vs_kappa_bnc[i] = throughput_per_ue_bnc[k]



            # Figure plot
            plt.plot(users, avg_throughput_jnt, label=r'\texttt{jnt}', marker='d')
            plt.plot(users, avg_throughput_seq, label=r'\texttt{seq}', marker='x')
            plt.plot(users, avg_throughput_bnc, label=r'\texttt{csi}', linestyle='dashed')
            filename = f'{n}_throughput_vs_K' + f'_kappa{kappa_dB:02.0f}'
            cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                          title=f'$\kappa = ${kappa_dB} [dB]', labels=['$K$', 'throughput [Mbit/s]'])

            plt.plot(users, throughput_per_ue_jnt, label=r'\texttt{jnt}', marker='d')
            plt.plot(users, throughput_per_ue_seq, label=r'\texttt{seq}', marker='x')
            plt.plot(users, throughput_per_ue_bnc, label=r'\texttt{csi}', linestyle='dashed')
            filename = f'{n}_throughput_per_ue_vs_K' + f'_kappa{kappa_dB:02.0f}'
            cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                          title=f'per user $\kappa = $ {kappa_dB} [dB]', labels=['$K$', 'throughput [Mbit/s]'])

            plt.plot(users, fairness_jnt, label=r'\texttt{jnt}', marker='d')
            plt.plot(users, fairness_seq, label=r'\texttt{seq}', marker='x')
            plt.plot(users, fairness_bnc, label=r'\texttt{csi}', linestyle='dashed')
            filename = f'{n}_fairness_vs_K' + f'_kappa{kappa_dB:02.0f}'
            cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                          title=f'$\kappa = ${kappa_dB} [dB]', labels=['$K$', "Jain's Fairness"])

        # Figure plot
        plt.plot(kappa_dB_vec, rate_vs_kappa_jnt, label=r'\texttt{jnt}', marker='d')
        plt.plot(kappa_dB_vec, rate_vs_kappa_seq, label='seq', marker='x')
        plt.plot(kappa_dB_vec, rate_vs_kappa_bnc, label=r'\texttt{csi}', linestyle='dashed')
        filename = f'{n}_throughput_vs_kappa'
        cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                      title=f'$K = ${users[k]}', labels=['$\kappa$', 'throughput [Mbit/s]'])



        plt.plot(kappa_dB_vec, fair_vs_kappa_jnt, label=r'\texttt{jnt}', marker='d')
        plt.plot(kappa_dB_vec, fair_vs_kappa_seq, label=r'\texttt{seq}', marker='x')
        plt.plot(kappa_dB_vec, fair_vs_kappa_bnc, label=r'\texttt{csi}', linestyle='dashed')
        filename = f'{n}_fairness_vs_kappa'
        cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                      title=f'$K = ${users[k]}', labels=['$\kappa$', "Jain's Fairness"])