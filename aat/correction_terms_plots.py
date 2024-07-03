import matplotlib.pyplot as plt
import numpy as np
import os

training_data_folder = '../aat/training_data/'

for file in os.listdir(training_data_folder):
    if 'correction_terms' in file:
        generator_idx = file.split('_')[1]
        data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)

        n_bins = int(0.05 * len(data))

        plt.grid()
        plt.hist(data, bins=n_bins, alpha=0.75)
        plt.xlabel('Correction Term')
        plt.ylabel('Counts')
        adjustment = '_enh' if '_enh' in file else ''
        plt.savefig(f'../aat/analysis/generator_{generator_idx}_correction_terms_distribution{adjustment}.png',
                    bbox_inches='tight')
        plt.clf()
