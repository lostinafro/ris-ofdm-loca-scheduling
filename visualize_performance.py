import os.path as path

import matplotlib.pyplot as plt
import numpy as np

import scenario.common as cmn
from environment import command_parser, OUTPUT_DIR, RISFrequencyEnv2D
from test_matching_max_rate import blk_dB_vec


bs_pos = np.array([[10, 100, 0]])
# Generate a dummy environment to get the number of std configuration
env = RISFrequencyEnv2D(5, bs_pos, np.array([[0,0,0]]))
Cmax = env.ris.num_std_configs

resources = 50

if __name__ == '__main__':
    # input parser
    render,  datadir = command_parser()

    names = ['max_rate', 'max_min']

    for blk_dB in blk_dB_vec:
        print(f'shadowing factor = {blk_dB}')

        for n in names:
            # Load data
            filename = path.join(datadir, f'RBs{resources:03d}_{n}_vsK_blk{blk_dB:02.0f}.npz')

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
                from test_matching_max_rate import ue_init, quant, overloading_factor
            else:
                from test_matching_max_min import ue_init, quant, overloading_factor

            users = np.arange(ue_init, overloading_factor * resources * Cmax + quant, quant)

            # Compute the average rate per user
            throughput_per_ue_jnt = np.mean(avg_throughput_jnt / ues_allocated_jnt, axis=0) / 1e6
            throughput_per_ue_seq = np.mean(avg_throughput_seq / ues_allocated_seq, axis=0) / 1e6
            throughput_per_ue_bnc = np.mean(avg_throughput_bnc / ues_allocated_bnc, axis=0) / 1e6

            # Take the average over the batch size
            avg_throughput_jnt = np.mean(avg_throughput_jnt, axis=0) / 1e6
            avg_throughput_seq = np.mean(avg_throughput_seq, axis=0) / 1e6
            avg_throughput_bnc = np.mean(avg_throughput_bnc, axis=0) / 1e6
            fairness_jnt = np.mean(fairness_jnt, axis=0)
            fairness_seq = np.mean(fairness_seq, axis=0)
            fairness_bnc = np.mean(fairness_bnc, axis=0)

            # Figure plot
            plt.plot(users, avg_throughput_jnt, label='RIS jnt', marker='d')
            plt.plot(users, avg_throughput_seq, label='RIS seq', marker='x')
            plt.plot(users, avg_throughput_bnc, label='No RIS', linestyle='dashed')
            filename = f'{n}_throughput_vs_K' + f'_blk{blk_dB:02.0f}'
            cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                          title=f'$B = ${blk_dB} [dB]', labels=['$K$', 'throughput [Mbit/s]'])

            plt.plot(users, throughput_per_ue_jnt, label='RIS jnt', marker='d')
            plt.plot(users, throughput_per_ue_seq, label='RIS seq', marker='x')
            plt.plot(users, throughput_per_ue_bnc, label='No RIS', linestyle='dashed')
            filename = f'{n}_throughput_per_ue_vs_K' + f'_blk{blk_dB:02.0f}'
            cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                          title=f'per user $B = $ {blk_dB} [dB]', labels=['$K$', 'throughput [Mbit/s]'])

            plt.plot(users, fairness_jnt, label='RIS jnt', marker='d')
            plt.plot(users, fairness_seq, label='RIS seq', marker='x')
            plt.plot(users, fairness_bnc, label='No RIS', linestyle='dashed')
            filename = f'{n}_fairness_vs_K' + f'_blk{blk_dB:02.0f}'
            cmn.printplot(render=render, filename=filename, dirname=OUTPUT_DIR,
                          title=f'$B = ${blk_dB} [dB]', labels=['$K$', "Jain's Fairness"])


