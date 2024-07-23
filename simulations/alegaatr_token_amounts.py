import os
import pandas as pd

FOLDER = '../simulations/stored_games_for_network_plots/'

file_list = os.listdir(FOLDER)

for file in file_list:
    df = pd.read_csv(f'{FOLDER}{file}')
    n_rounds = df.shape[0] - 1
    player_cols = [col for col in df.columns if 'p' in col and col[1].isdigit() and '-' not in col]
    n_tokens, alegaatr_idx = 2 * len(player_cols), player_cols[-1]
    n_keep, n_give, n_steal = 0, 0, 0

    for rnd in range(n_rounds):
        for player_idx in player_cols:
            alegaatr_to_player_tokens = df.loc[df.index[rnd], f'{alegaatr_idx}-T-{player_idx}'] * n_tokens

            if alegaatr_to_player_tokens > 0:
                if player_idx == alegaatr_idx:
                    n_keep += alegaatr_to_player_tokens

                else:
                    n_give += alegaatr_to_player_tokens

            elif alegaatr_to_player_tokens < 0:
                assert player_idx != alegaatr_idx
                n_steal += abs(alegaatr_to_player_tokens)

    print(file)
    print(f'Keep: {n_keep}')
    print(f'Give: {n_give}')
    print(f'Steal: {n_steal}')
    if n_give > n_keep or n_steal > n_keep:
        print('GAVE MORE')
    print()
