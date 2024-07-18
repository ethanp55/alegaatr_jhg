import numpy as np
import os


def _calculate_percentile(array, value):
    sorted_array = np.sort(array)
    rank = np.searchsorted(sorted_array, value, side='right')

    return rank / len(array)


# gen_names = [f'generator_{i}' for i in range(16)]
gen_names = ['generator_10']
agent_names = ['AlegAATr', 'EXP4', 'EEE', 'UCB', 'D-UCB', 'R-UCB', 'SW-UCB', 'DQN'] + gen_names
folder = '../simulations/results/'

for agent_name in agent_names:
    n_simulations, n_wins = 0, 0
    percentiles, pop_sums, pops = [], [], []
    agent_files = [file for file in os.listdir(folder) if file[:len(agent_name)] == agent_name and file[
        len(agent_name)] == '_']  # Only look at agent-specific files

    for file in agent_files:
        data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)
        data = [data] if len(data.shape) == 1 else data

        for row in data:
            pop = row[-1]
            pops.append(pop)
            percentiles.append(_calculate_percentile(row, pop))
            pop_sums.append(sum(row))
            max_pop = row.max()
            n_wins += 1 if pop == max_pop else 0
            n_simulations += 1

    print(f'Results for {agent_name}: ')
    print(f'Num simulations: {n_simulations}')
    print(f'Num wins: {n_wins}')
    print(f'Win rate: {round(n_wins / n_simulations, 3)}')
    print(f'Average percentile: {sum(percentiles) / len(percentiles)}')
    print(f'Average pop sum: {sum(pop_sums) / len(pop_sums)}')
    print(f'Average pop: {sum(pops) / len(pops)}\n')
