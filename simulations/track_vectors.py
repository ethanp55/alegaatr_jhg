from aat.train_generators import cabs_with_random_params, random_selection_of_best_trained_cabs, random_agents, \
    basic_bandits, random_mixture_of_all_types, create_society
from copy import deepcopy
from functools import partial
from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.dqn import DQNAgent
from GeneSimulation_py.ducb import DUCB
from GeneSimulation_py.eee import EEE
from GeneSimulation_py.exp4 import EXP4
from GeneSimulation_py.main import run_with_specified_agents
from GeneSimulation_py.rucb import RUCB
from GeneSimulation_py.swucb import SWUCB
from GeneSimulation_py.ucb import UCB
from multiprocessing import Process
import numpy as np
from typing import List


def society_of_bandits(max_players: int = 20) -> List[AbstractAgent]:
    bandits = [AlegAATr(lmbda=0.0, ml_model_type='knn', enhanced=True), EXP4(), EEE(), UCB(), DUCB(), RUCB(), SWUCB(),
               DQNAgent(train_network=False)]
    agents = []

    for _ in range(max_players):
        bandit = np.random.choice(bandits)
        agents.append(deepcopy(bandit))

    return agents


def self_play(agent: AbstractAgent, max_players: int = 20) -> List[AbstractAgent]:
    agent_copies = [deepcopy(agent) for _ in range(max_players)]
    if isinstance(agent, AlegAATr):
        for copy in agent_copies:
            copy.generator_usage_file = None
    return agent_copies


N_EPOCHS = 5
INITIAL_POP_CONDITIONS = ['equal']
N_PLAYERS = [10]
N_ROUNDS = [20]
N_CATS = [0, 1, 2]

names = []


def simulations() -> None:
    # Variables to track progress
    n_iterations = N_EPOCHS * len(INITIAL_POP_CONDITIONS) * len(N_PLAYERS) * len(N_ROUNDS) * len(N_CATS)
    progress_percentage_chunk = int(0.05 * n_iterations)
    curr_iteration = 0
    print(n_iterations, progress_percentage_chunk)

    # Run the simulation process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    for n_cats in N_CATS:
                        if curr_iteration != 0 and progress_percentage_chunk != 0 and \
                                curr_iteration % progress_percentage_chunk == 0:
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
                        list_of_opponents.append((random_mixture_of_all_types(n_other_players), 'mixture'))
                        list_of_opponents.append((society_of_bandits(n_other_players), 'banditsociety'))
                        list_of_opponents.append(([], 'selfplay'))

                        for opponents, opponents_label in list_of_opponents:
                            # Create different agents to test
                            agents_to_test = []
                            agents_to_test.append(AlegAATr(lmbda=0.0, ml_model_type='knn', enhanced=True,
                                                           track_vector_file=f'../simulations/vectors/AlegAATr_{opponents_label}_pop={initial_pop_condition}_p={n_players}_r={n_rounds}_c={n_cats}'))
                            agents_to_test.append(DQNAgent(train_network=False,
                                                           track_vector_file=f'../simulations/vectors/DQN_{opponents_label}_pop={initial_pop_condition}_p={n_players}_r={n_rounds}_c={n_cats}'))

                            for agent_to_test in agents_to_test:
                                # Create cats (if any)
                                cats = [AssassinAgent() for _ in range(n_cats)]
                                opps = self_play(agent_to_test,
                                                 n_other_players) if opponents_label == 'selfplay' else deepcopy(
                                    opponents)
                                players = create_society(agent_to_test, cats, opps, n_players)
                                partial_func = partial(run_with_specified_agents, players=players,
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
