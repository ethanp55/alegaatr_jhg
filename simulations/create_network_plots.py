from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from generator_analysis.network import NodeNetwork
import os
import pandas as pd
from PIL import Image, ImageOps

FOLDER = '../simulations/stored_games_for_network_plots/'


def create_plots() -> None:
    def _load_data(df):
        plrs = len([col for col in df.columns if col.startswith('p') and 2 <= len(col) <= 3])
        pop_cols = [f'p{i}' for i in range(plrs)]
        action_cols = [f'p{i}-T-p{j}' for i in range(plrs) for j in range(plrs)]
        infl_cols = [f'p{i}-I-p{j}' for i in range(plrs) for j in range(plrs)]
        pop_mat = df[pop_cols].to_numpy()
        actions_mat = df[action_cols].to_numpy().reshape(-1, plrs, plrs)
        infl_mat = df[infl_cols].to_numpy().reshape(-1, plrs, plrs)
        params = {
            'alpha': df['alpha'][0],
            'beta': df['beta'][0],
            'give': df['give'][0],
            'keep': df['keep'][0],
            'steal': df['steal'][0],
            'num_players': plrs
        }

        return pop_mat, actions_mat, infl_mat, params

    def _plot_game(df: pd.DataFrame, file_name: str, rounds=range(1, 31, 6)):
        pop_mat, actions_mat, infl_mat, params = _load_data(df)

        dpi = 200
        fig = plt.figure(figsize=(10, 6), dpi=dpi)

        names = [col for col in df.columns if col.startswith('p') and 2 <= len(col) <= 3]
        n_players = len(names)
        name2color = {}
        for i, name in enumerate(names):
            if i == len(names) - 1:
                name2color[name] = 'black'
            # if n_players - i <= N_GENERATORS:
            #     name2color[name] = 'black'
        legend_colors = None

        net = NodeNetwork()
        net.setupPlayers(names)
        net.initNodes(init_pops=pop_mat[0])

        gs = GridSpec(2, len(rounds), figure=fig)
        ids = [(1, i) for i in range(len(rounds))]
        for r in range(pop_mat.shape[0]):
            net.update(infl_mat[r], pop_mat[r])
            if r in rounds:
                round_idx = rounds.index(r)
                ax = fig.add_subplot(gs[ids[round_idx][0], ids[round_idx][1]], facecolor='c' if r % 2 == 1 else 'm',
                                     ymargin=-.4)
                net.graphExchange(ax, fig, actions_mat[r], color_lookup=name2color)
                ax.set_title(f'Round {r}', fontsize=12, loc='center', y=-0.25)
        ax = fig.add_subplot(gs[0, :])
        net._graphPopularities(ax, fig, pop_mat, color_lookup=name2color, legend_colors=legend_colors)
        fig.subplots_adjust(wspace=0.0, hspace=0.06)

        if 'AlegAATr' in file_name:
            generator_usage_file = file_name[9:]
            full_path = f'../simulations/alegaatr_generator_usage/{generator_usage_file}'
            generator_usage_df = pd.read_csv(full_path) if generator_usage_file in os.listdir(
                '../simulations/alegaatr_generator_usage/') else None

            if generator_usage_df is not None:
                for _, row in generator_usage_df.iterrows():
                    round_used, generator_at_round = row['round'], row['generator']
                    ax.axvline(x=round_used, color='black', linestyle='--')
                    ax.text(round_used, ax.get_ylim()[1], generator_at_round, ha='center', va='bottom', color='black')

        image_path = f'./network_plots/{file_name.split(".")[0]}.png'
        plt.savefig(image_path)

        image = Image.open(image_path)

        gray_image = ImageOps.grayscale(image)

        threshold_value = 254  # Adjust this value as needed
        thresholded_image = gray_image.point(lambda p: p < threshold_value and 255)
        bbox = thresholded_image.getbbox()

        cropped_image = image.crop(bbox)

        cropped_image.save(image_path)

    file_list = os.listdir(FOLDER)

    for file in file_list:
        # if 'AlegAATr_cabsrandomparams_pop=equal_p=15_r=40_c=0' not in file:
        #     continue
        df = pd.read_csv(f'{FOLDER}{file}')
        n_rounds = df.shape[0] - 1
        if n_rounds == 10:
            step_size = 3
        else:
            step_size = 10
        _plot_game(df, file, range(1, n_rounds + 1, step_size))


if __name__ == "__main__":
    create_plots()
