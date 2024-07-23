import pandas as pd

df = pd.read_csv('../simulations/stored_games_for_network_plots/generators_pop=equal_p=16_r=40_c=0.csv')
n_keep, n_give, n_steal = {}, {}, {}
transaction_columns = [col for col in df.columns if '-T-' in col]
player_cols = [col for col in df.columns if 'p' in col and col[1].isdigit() and '-' not in col]
n_tokens = 2 * (len(transaction_columns) ** 0.5)

for rnd in range(df.shape[0] - 1):
    for col in transaction_columns:
        player_a, player_b = col.split('-T')[0], col.split('T-')[1]
        a_to_b_tokens = df.loc[df.index[rnd], col] * n_tokens

        if a_to_b_tokens > 0:
            if player_a == player_b:
                n_keep[player_a] = n_keep.get(player_a, 0) + a_to_b_tokens

            else:
                n_give[player_a] = n_give.get(player_a, 0) + a_to_b_tokens

        elif a_to_b_tokens < 0:
            assert player_a != player_b
            n_steal[player_a] = n_steal.get(player_a, 0) + abs(a_to_b_tokens)

for player in player_cols:
    keep, give, steal = n_keep.get(player, 0), n_give.get(player, 0), n_steal.get(player, 0)
    print(player)
    print(f'Keep: {keep}')
    print(f'Give: {give}')
    print(f'Steal: {steal}')
    if give > keep or steal > keep:
        print(f'HERE - {player}')
    print()
