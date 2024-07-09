import numpy as np
import pandas as pd
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd

MAX_N_PLAYERS = 20


def _calculate_percentile(array, value):
    sorted_array = np.sort(array)
    rank = np.searchsorted(sorted_array, value, side='right')

    return rank / len(array)


results, folder = {}, '../simulations/results/'

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if agent_name == 'DQN':  # TODO: remove this
        continue
    pop_condition = file.split('pop=')[1].split('_')[0]
    n_players = file.split('p=')[2].split('_')[0]
    n_rounds = file.split('r=')[1].split('_')[0]
    n_cats = file.split('c=')[1].split('_')[0][0]

    # if f'pop={pop_condition}' not in results:
    #     results[f'pop={pop_condition}'] = {}
    if f'p={n_players}' not in results:
        results[f'p={n_players}'] = {}
    if f'r={n_rounds}' not in results:
        results[f'r={n_rounds}'] = {}
    if f'c={n_cats}' not in results:
        results[f'c={n_cats}'] = {}
    if 'overall' not in results:
        results['overall'] = {}

    data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)
    n_rows, n_cols = data.shape
    if n_cols < MAX_N_PLAYERS:
        n_zeroes = MAX_N_PLAYERS - n_cols
        zeroes = np.zeros((n_rows, n_zeroes))
        data = np.concatenate((zeroes, data), axis=1)

    # results[f'pop={pop_condition}'][agent_name] = np.concatenate(
    #     (results[f'pop={pop_condition}'][agent_name], data)) if agent_name in results[f'pop={pop_condition}'] else data
    results[f'p={n_players}'][agent_name] = np.concatenate(
        (results[f'p={n_players}'][agent_name], data)) if agent_name in results[f'p={n_players}'] else data
    results[f'r={n_rounds}'][agent_name] = np.concatenate(
        (results[f'r={n_rounds}'][agent_name], data)) if agent_name in results[f'r={n_rounds}'] else data
    results[f'c={n_cats}'][agent_name] = np.concatenate(
        (results[f'c={n_cats}'][agent_name], data)) if agent_name in results[f'c={n_cats}'] else data
    results['overall'][agent_name] = np.concatenate(
        (results['overall'][agent_name], data)) if agent_name in results['overall'] else data

for scenario_str, scenario_results in results.items():
    names, pop_sums, final_pops, percentiles, wins = [], [], [], [], []

    for agent_name, agent_results in scenario_results.items():
        for row in agent_results:
            agent_pop = row[-1]
            names.append(agent_name)
            pop_sums.append(sum(row))
            final_pops.append(agent_pop)
            percentiles.append(_calculate_percentile(row, agent_pop))
            wins.append(1 if agent_pop == row.max() else 0)

    print('-----------------------------------------------')
    print(f'STATISTICAL TESTS FOR {scenario_str} SCENARIOS')
    print('-----------------------------------------------')

    print('FINAL POPS:')
    df = pd.DataFrame({'algorithm': names, 'results': final_pops})
    print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    print('\nPERCENTILES:')
    df = pd.DataFrame({'algorithm': names, 'results': percentiles})
    print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    print('\nWINS:')
    df = pd.DataFrame({'algorithm': names, 'results': wins})
    print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    print('\nPOP SUMS:')
    df = pd.DataFrame({'algorithm': names, 'results': pop_sums})
    print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print('-----------------------------------------------\n')
