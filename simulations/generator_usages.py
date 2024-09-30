import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

folder = '../simulations/generator_usage/'
files = os.listdir(folder)
agent_names = set()
N_GENERATORS = 8

for file in files:
    agent_name = file.split('_')[0]
    agent_names.add(agent_name)

for agent_name in agent_names:
    pop_conditions, num_players, num_rounds, num_cats, opponent_types = [], [], [], [], []
    list_of_generator_counts = [[] for _ in range(N_GENERATORS)]

    for file in files:
        name = file.split('_')[0]
        if name != agent_name:
            continue
        pop_condition = file.split('pop=')[1].split('_')[0]
        n_players = file.split('p=')[2].split('_')[0]
        n_rounds = file.split('r=')[1].split('_')[0]
        n_cats = file.split('c=')[1].split('_')[0][0]
        opp_type = file[len(agent_name) + 1:].split('_')[0]
        opp_type = 'banditsociety' if opp_type == 'basicbandits' else opp_type
        data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=1)
        curr_generator_counts = {}

        for row in data:
            generator = row[-1]
            curr_generator_counts[generator] = curr_generator_counts.get(generator, 0) + 1

        for i in range(N_GENERATORS):
            gen_count = curr_generator_counts.get(i, 0)
            list_of_generator_counts[i].append(gen_count)

        pop_conditions.append(pop_condition)
        num_players.append(n_players)
        num_rounds.append(n_rounds)
        num_cats.append(n_cats)
        opponent_types.append(opp_type)

    data_dict = {
        'pop_condition': pop_conditions,
        'n_players': num_players,
        'n_rounds': num_rounds,
        'n_cats': num_cats,
        'opponent_type': opponent_types,
    }
    for i in range(N_GENERATORS):
        data_dict[f'generator_{i}'] = list_of_generator_counts[i]

    df = pd.DataFrame(data_dict)

    # Plot generator usages by number of rounds
    df_melted = pd.melt(df, id_vars=['n_rounds'], value_vars=[f'generator_{i}' for i in range(N_GENERATORS)],
                        var_name='generator', value_name='usage_count')
    df_grouped = df_melted.groupby(['n_rounds', 'generator']).sum().reset_index()
    unique_num_rounds = df_grouped['n_rounds'].unique()
    bar_positions = np.arange(len(df_grouped['generator'].unique()))
    bar_width = 0.2
    fig, ax = plt.subplots()
    for i, num_rounds in enumerate(unique_num_rounds):
        subset = df_grouped[df_grouped['n_rounds'] == num_rounds]
        ax.bar(bar_positions + i * bar_width, subset['usage_count'], bar_width, label=f'rounds={num_rounds}')
    ax.set_xticks(bar_positions + bar_width * (len(unique_num_rounds) - 1) / 2)
    ax.set_xticklabels([i for i in range(N_GENERATORS)])
    ax.set_xlabel('Generator')
    ax.set_ylabel('Usage Count')
    ax.legend()
    plt.grid()
    plt.savefig(f'../simulations/generator_usage_plots/{agent_name}_rounds.png', bbox_inches='tight')
    plt.clf()

    # Plot generator usages by number of players
    df_melted = pd.melt(df, id_vars=['n_players'], value_vars=[f'generator_{i}' for i in range(N_GENERATORS)],
                        var_name='generator', value_name='usage_count')
    df_grouped = df_melted.groupby(['n_players', 'generator']).sum().reset_index()
    unique_num_players = df_grouped['n_players'].unique()
    bar_positions = np.arange(len(df_grouped['generator'].unique()))
    bar_width = 0.2
    fig, ax = plt.subplots()
    for i, num_players in enumerate(unique_num_players):
        subset = df_grouped[df_grouped['n_players'] == num_players]
        ax.bar(bar_positions + i * bar_width, subset['usage_count'], bar_width, label=f'players={num_players}')
    ax.set_xticks(bar_positions + bar_width * (len(unique_num_players) - 1) / 2)
    ax.set_xticklabels([i for i in range(N_GENERATORS)])
    ax.set_xlabel('Generator')
    ax.set_ylabel('Usage Count')
    ax.legend()
    plt.grid()
    plt.savefig(f'../simulations/generator_usage_plots/{agent_name}_players.png', bbox_inches='tight')
    plt.clf()

    # Plot generator usages by number of CATs
    df_melted = pd.melt(df, id_vars=['n_cats'], value_vars=[f'generator_{i}' for i in range(N_GENERATORS)],
                        var_name='generator', value_name='usage_count')
    df_grouped = df_melted.groupby(['n_cats', 'generator']).sum().reset_index()
    unique_num_cats = df_grouped['n_cats'].unique()
    bar_positions = np.arange(len(df_grouped['generator'].unique()))
    bar_width = 0.2
    fig, ax = plt.subplots()
    for i, num_cats in enumerate(unique_num_cats):
        subset = df_grouped[df_grouped['n_cats'] == num_cats]
        ax.bar(bar_positions + i * bar_width, subset['usage_count'], bar_width, label=f'cats={num_cats}')
    ax.set_xticks(bar_positions + bar_width * (len(unique_num_cats) - 1) / 2)
    ax.set_xticklabels([i for i in range(N_GENERATORS)])
    ax.set_xlabel('Generator')
    ax.set_ylabel('Usage Count')
    ax.legend()
    plt.grid()
    plt.savefig(f'../simulations/generator_usage_plots/{agent_name}_cats.png', bbox_inches='tight')
    plt.clf()
