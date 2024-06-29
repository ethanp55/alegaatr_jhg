from aat.train_generators import BasicBandit, FavorMoreRecent, UniformSelector
from aat.train_generators import cabs_with_random_params, random_selection_of_best_trained_cabs, random_agents, \
    basic_bandits, random_mixture_of_all_types, create_society
from copy import deepcopy
from functools import partial
from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.eee import EEE
from GeneSimulation_py.exp4 import EXP4
from GeneSimulation_py.main import run_with_specified_agents
from GeneSimulation_py.ucb import UCB
from multiprocessing import Process
import os
from typing import List


def uniform_selectors(max_players: int = 20) -> List[AbstractAgent]:
    return [UniformSelector() for _ in range(max_players)]


def favor_more_recents(max_players: int = 20) -> List[AbstractAgent]:
    return [FavorMoreRecent() for _ in range(max_players)]


N_EPOCHS = 2
INITIAL_POP_CONDITIONS = ['equal']
N_PLAYERS = [5, 10, 15]
N_ROUNDS = [20, 30, 40]
N_CATS = [0, 1]


def simulations() -> None:
    # Variables to track progress
    n_iterations = N_EPOCHS * len(INITIAL_POP_CONDITIONS) * len(N_PLAYERS) * len(N_ROUNDS) * len(N_CATS)
    progress_percentage_chunk = int(0.05 * n_iterations)
    curr_iteration = 0
    print(n_iterations, progress_percentage_chunk)

    # Reset any existing simulation files (opening a file in write mode will truncate it)
    for file in os.listdir('../simulations/results/'):
        if 'AlegAATr' in file:
            with open(f'../simulations/results/{file}', 'w', newline='') as _:
                pass

    # Run the simulation process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    for n_cats in N_CATS:
                        if curr_iteration != 0 and curr_iteration % progress_percentage_chunk == 0:
                            print(f'{100 * (curr_iteration / n_iterations)}%')
                        list_of_sims_to_run = []

                        # Create players, aside from main agent to test and any cats
                        n_other_players = n_players - 1 - n_cats
                        list_of_opponents = []
                        list_of_opponents.append((cabs_with_random_params(n_other_players), 'cabsrandomparams'))
                        list_of_opponents.append((random_selection_of_best_trained_cabs('../ResultsSaved/no_cat/',
                                                                                        n_other_players),
                                                  'cabsnocat'))
                        list_of_opponents.append((random_selection_of_best_trained_cabs('../ResultsSaved/one_cat/',
                                                                                        n_other_players),
                                                  'cabsonecat'))
                        list_of_opponents.append((random_selection_of_best_trained_cabs('../ResultsSaved/two_cats/',
                                                                                        n_other_players),
                                                  'cabstwocats'))
                        list_of_opponents.append((random_agents(n_other_players), 'randoms'))
                        list_of_opponents.append((basic_bandits(max_players=n_other_players), 'basicbandits'))
                        # list_of_opponents.append((uniform_selectors(n_other_players), 'uniformselectors'))
                        # list_of_opponents.append((favor_more_recents(n_other_players), 'favormorerecents'))
                        list_of_opponents.append((random_mixture_of_all_types(n_other_players), 'mixture'))

                        for opponents, opponents_label in list_of_opponents:
                            # Create different agents to test
                            agents_to_test = []
                            agents_to_test.append(AlegAATr(lmbda=0.0, ml_model_type='knn'))
                            # agents_to_test.append(EXP4())
                            # agents_to_test.append(EEE())
                            # agents_to_test.append(UCB())
                            # agents_to_test.append(BasicBandit())
                            # agents_to_test.append(UniformSelector())
                            # agents_to_test.append(FavorMoreRecent())

                            for agent_to_test in agents_to_test:
                                # Create cats (if any)
                                cats = [AssassinAgent() for _ in range(n_cats)]
                                players = create_society(agent_to_test, cats, deepcopy(opponents), n_players)
                                simulation_label = f'{agent_to_test.whoami}_{opponents_label}_pop={initial_pop_condition}_p={n_players}_r={n_rounds}_c={n_cats}'
                                partial_func = partial(run_with_specified_agents, players=players,
                                                       final_pops_file=f'../simulations/results/{simulation_label}.csv',
                                                       initial_pop_setting=initial_pop_condition, numRounds=n_rounds)
                                list_of_sims_to_run.append(partial_func)

                        # Spin off a process for each grouping of players and play the game
                        processes = []

                        for sim_func in list_of_sims_to_run:
                            process = Process(target=sim_func)
                            processes.append(process)
                            process.start()

                        for process in processes:
                            process.join()  # Wait for every process to finish before continuing

                        curr_iteration += 1


if __name__ == "__main__":
    simulations()