from aat.train_generators import cabs_with_random_params, random_selection_of_best_trained_cabs, random_agents, \
    basic_bandits, random_mixture_of_all_types, create_society
from aat.train_generators import N_EPOCHS, INITIAL_POP_CONDITIONS, N_PLAYERS, N_ROUNDS
from copy import deepcopy
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.dqn import DQNAgent
from GeneSimulation_py.main import run_with_specified_agents
import numpy as np

# Variables to track progress
N_EPOCHS_ADJ = N_EPOCHS * 10
n_training_iterations = N_EPOCHS_ADJ * len(INITIAL_POP_CONDITIONS) * len(N_PLAYERS) * len(N_ROUNDS)
progress_percentage_chunk = int(0.02 * n_training_iterations)
curr_iteration = 0
print(n_training_iterations, progress_percentage_chunk)

# Variables for cats
np.random.seed(42)  # For consistency in selecting number of cat agents
n_cats_list = list(np.random.choice(a=[0, 1, 2], size=n_training_iterations, p=[0.9, 0.05, 0.05]))
print(f'CATS: {n_cats_list}')
np.random.seed()
cat_idx = 0

# Deep Q-learner
dqn = DQNAgent(train_network=True)

# Run the training process
for epoch in range(N_EPOCHS_ADJ):
    print(f'Epoch {epoch + 1}')

    # Keep track of win rate
    n_wins, n_games = 0, 0

    for initial_pop_condition in INITIAL_POP_CONDITIONS:
        for n_players in N_PLAYERS:
            for n_rounds in N_ROUNDS:
                n_cats = n_cats_list[cat_idx]
                print(f'N CATS: {n_cats}')
                cat_idx += 1
                if curr_iteration != 0 and curr_iteration % progress_percentage_chunk == 0:
                    print(f'{100 * (curr_iteration / n_training_iterations)}%')
                list_of_players = []

                # Create players, aside from main agent to train on and any cats
                n_other_players = n_players - 1 - n_cats
                list_of_opponents = []
                list_of_opponents.append(cabs_with_random_params(n_other_players))
                list_of_opponents.append(
                    random_selection_of_best_trained_cabs('../ResultsSaved/no_cat/', n_other_players))
                list_of_opponents.append(
                    random_selection_of_best_trained_cabs('../ResultsSaved/one_cat/', n_other_players))
                list_of_opponents.append(
                    random_selection_of_best_trained_cabs('../ResultsSaved/two_cats/', n_other_players))
                list_of_opponents.append(random_agents(n_other_players))
                list_of_opponents.append(basic_bandits(max_players=n_other_players))
                list_of_opponents.append(random_mixture_of_all_types(n_other_players))

                for opponents in list_of_opponents:
                    cats = [AssassinAgent() for _ in range(n_cats)]
                    players = create_society(dqn, cats, deepcopy(opponents), n_players)

                    # Run the game
                    final_pops = run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                           numRounds=n_rounds)
                    dqn_pop = final_pops[-1]
                    n_wins += 1 if dqn_pop == final_pops.max() else 0
                    n_games += 1

                    # At the end of each training simulation, reset import parameters for the Q-learner
                    dqn.reset()

                curr_iteration += 1

    # At the end of each epoch/episode
    dqn.train()
    dqn.update_epsilon()
    dqn.update_networks()
    dqn.clear_buffer()

    print(f'Win rate: {n_wins / n_games}')
