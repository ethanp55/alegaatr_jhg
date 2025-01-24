from aat.train_generators import FavorMoreRecent, UniformSelector
from aat.train_generators import cabs_with_random_params, random_selection_of_best_trained_cabs, random_agents, \
    basic_bandits, random_mixture_of_all_types, create_society
from copy import deepcopy
from functools import partial
from GeneSimulation_py.aalegaatr import AAlegAATr
from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.aleqgaatr import AleqgAATr
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.dqn import DQNAgent
from GeneSimulation_py.ducb import DUCB
from GeneSimulation_py.eee import EEE
from GeneSimulation_py.exp4 import EXP4
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.madqn import MADQN
from GeneSimulation_py.main import run_with_specified_agents
from GeneSimulation_py.qalegaatr import QAlegAATr
from GeneSimulation_py.raat import RAAT
from GeneSimulation_py.ralegaatr import RAlegAATr
from GeneSimulation_py.rawo import RawO
from GeneSimulation_py.rdqn import RDQN
from GeneSimulation_py.rucb import RUCB
from GeneSimulation_py.smalegaatr import SMAlegAATr
from GeneSimulation_py.soaleqgaatr import SOAleqgAATr
from GeneSimulation_py.swucb import SWUCB
from GeneSimulation_py.ucb import UCB
from multiprocessing import Process
import numpy as np
import os
import pandas as pd
from typing import List
from GeneSimulation_py.generator_pool import GeneratorPool
from utils.utils import BASELINE
from aat.knn import fit_knn_models
from aat.train_raat import train_raat
from aat.train_rawo import train_raw
from aat.train_qalegaatr import train_qalegaatr
from GeneSimulation_py.train_dqn import train_dqn
from GeneSimulation_py.train_ralegaatr import train_ralegaatr
from GeneSimulation_py.train_aleqgaatr import train_aleqgaatr
from adaptability_sims import adaptability


# Simple bandit agent that periodically explores and exploits otherwise
class BasicBandit(AbstractAgent):
    def __init__(self, epsilon: float = 0.1, epsilon_decay: float = 0.99, check_assumptions: bool = False) -> None:
        super().__init__()
        self.whoami = 'BasicBandit'
        self.epsilon, self.epsilon_decay = epsilon, epsilon_decay
        self.generator_pool = GeneratorPool(check_assumptions=check_assumptions)
        self.check_assumptions = check_assumptions
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.empirical_increases = {}
        self.prev_popularity = None

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        if self.check_assumptions:
            self.generator_pool.train_aat(player_idx, round_num, received, popularities, influence, extra_data, v,
                                          transactions, self.generator_to_use_idx, BASELINE)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        curr_popularity = popularities[player_idx]

        # Update empirical rewards
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.empirical_increases[self.generator_to_use_idx] = \
                self.empirical_increases.get(self.generator_to_use_idx, []) + [increase]
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)
        # Explore
        if np.random.rand() < self.epsilon:
            self.generator_to_use_idx = np.random.choice(self.generator_indices)

        # Exploit
        else:
            best_i, best_avg_increase = None, -np.inf

            for i in self.generator_indices:
                increases = self.empirical_increases.get(i, [])

                # If the generator hasn't been used yet, try it
                if len(increases) == 0:
                    best_i = i
                    break

                avg_increase = sum(increases) / len(increases)

                if avg_increase > best_avg_increase:
                    best_i, best_avg_increase = i, best_avg_increase

            self.generator_to_use_idx = best_i

        # Slowly decrease the probability of exploring
        self.epsilon *= self.epsilon_decay

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations


# Agent that just randomly (uniform) chooses a generator to use
class UniformSelector(AbstractAgent):
    def __init__(self, check_assumptions: bool = False, no_baseline: bool = False) -> None:
        super().__init__()
        self.whoami = 'UniformSelector'
        self.generator_pool = GeneratorPool(check_assumptions=check_assumptions, no_baseline_labels=no_baseline)
        self.check_assumptions = check_assumptions
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        if self.check_assumptions:
            self.generator_pool.train_aat(player_idx, round_num, received, popularities, influence, extra_data, v,
                                          transactions, self.generator_to_use_idx, BASELINE)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Randomly (uniform) choose a generator to use
        self.generator_to_use_idx = np.random.choice(self.generator_indices)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations


# Agent that favors generators that have been used more recently
class FavorMoreRecent(AbstractAgent):
    def __init__(self, check_assumptions: bool = False, no_baseline: bool = False) -> None:
        super().__init__()
        self.whoami = 'FavorMoreRecent'
        self.generator_pool = GeneratorPool(check_assumptions=check_assumptions, no_baseline_labels=no_baseline)
        self.check_assumptions = check_assumptions
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx, self.prev_generator_idx = None, None
        self.n_rounds_since_last_use = {}
        self.max_in_a_row = 5
        self.n_rounds_used = 0

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        if self.check_assumptions:
            self.generator_pool.train_aat(player_idx, round_num, received, popularities, influence, extra_data, v,
                                          transactions, self.generator_to_use_idx, BASELINE)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Randomly choose a generator, but favor those that have been used most recently
        rounds_since_used = [1 / self.n_rounds_since_last_use.get(i, 1) for i in self.generator_indices]
        if self.prev_generator_idx is not None and self.prev_generator_idx == self.generator_to_use_idx and \
                self.n_rounds_used >= self.max_in_a_row:
            rounds_since_used[self.generator_to_use_idx] = 0
            self.n_rounds_used = 0
        sum_val = sum(rounds_since_used)

        probabilities = [x / sum_val for x in rounds_since_used]
        self.prev_generator_idx = self.generator_to_use_idx
        self.generator_to_use_idx = np.random.choice(self.generator_indices, p=probabilities)

        # Update the number of rounds since each generator was used
        for i in self.generator_indices:
            self.n_rounds_since_last_use[i] = (
                    self.n_rounds_since_last_use.get(i, 1) + 1) if i != self.generator_to_use_idx else 1

        self.n_rounds_used += 1

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations


# Agent that just randomly (uniform) chooses a generator to use for the entire game
class RandomAgent(AbstractAgent):
    def __init__(self) -> None:
        super().__init__()
        self.whoami = 'random'
        generator_pool = GeneratorPool()
        self.generator = np.random.choice(generator_pool.generators)

    def setGameParams(self, game_params, forced_random) -> None:
        self.generator.setGameParams(game_params, forced_random)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        token_allocations = self.generator.play_round(player_idx, round_num, received, popularities, influence,
                                                      extra_data, v, transactions)

        self.generator.update_prev_allocations(token_allocations)

        return token_allocations


# ----------------------------------------------------------------------------------------------------------------------
# Functions for creating the society of players ------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def cabs_with_random_params(max_players: int = 20) -> List[AbstractAgent]:
    # A CAB will choose random parameters if the gene string is empty ('')
    return [GeneAgent3('', 1) for _ in range(max_players)]


def random_selection_of_best_trained_cabs(folder: str, max_players: int = 20) -> List[AbstractAgent]:
    cabs = []
    files = os.listdir(folder)
    # Choose from one of the final 50 generations
    files_filtered = [file for file in files if int(file.split('_')[1].split('.')[0]) >= 150]
    file_to_use = np.random.choice(files_filtered)

    df = pd.read_csv(f'{folder}{file_to_use}', header=None)

    for i in range(max_players):
        gene_str = df.iloc[i, 0]
        cabs.append(GeneAgent3(gene_str, 1))

    return cabs


def random_agents(max_players: int = 20) -> List[AbstractAgent]:
    return [RandomAgent() for _ in range(max_players)]


def basic_bandits(epsilon: float = 0.1, epsilon_decay: float = 0.99, max_players: int = 20) -> List[AbstractAgent]:
    return [BasicBandit(epsilon, epsilon_decay) for _ in range(max_players)]


def random_mixture_of_all_types(max_players: int = 20) -> List[AbstractAgent]:
    agents = []
    agents.extend(cabs_with_random_params(max_players))
    agents.extend(random_selection_of_best_trained_cabs('../ResultsSaved/no_cat/', max_players))
    agents.extend(random_selection_of_best_trained_cabs('../ResultsSaved/one_cat/', max_players))
    agents.extend(random_selection_of_best_trained_cabs('../ResultsSaved/two_cats/', max_players))
    agents.extend(random_agents(max_players))
    agents.extend(basic_bandits(max_players=max_players))

    np.random.shuffle(agents)

    return agents[:max_players]


def create_society(our_player: AbstractAgent, cats: List[AssassinAgent], all_other_players: List[AbstractAgent],
                   n_players: int) -> List[AbstractAgent]:
    players = []
    np.random.shuffle(all_other_players)

    for i in range(n_players - len(cats) - 1):
        players.append(all_other_players[i])

    for cat in cats:
        players.append(cat)

    players.append(our_player)

    assert len(players) == n_players

    return players


N_TRAIN_TEST_RUNS = 5
N_EPOCHS = 5
INITIAL_POP_CONDITIONS = ['equal', 'random']
N_PLAYERS = [5, 10, 15]
N_ROUNDS = [20, 30, 40]

# Reset any existing simulation files (opening a file in write mode will truncate it)
for file in os.listdir('../simulations/adaptability_results/'):
    with open(f'../simulations/adaptability_results/{file}', 'w', newline='') as _:
        pass

for run_num in range(N_TRAIN_TEST_RUNS):
    print(f'RUN NUM = {run_num + 1}')

    print('Training REGAETune agents...')

    # Train AlegAATr
    NO_BASELINE = False

    n_training_iterations = N_EPOCHS * len(INITIAL_POP_CONDITIONS) * len(N_PLAYERS) * len(N_ROUNDS)
    np.random.seed(42)
    n_cats_list = list(np.random.choice(a=[0, 1, 2], size=n_training_iterations, p=[0.9, 0.05, 0.05]))
    print(f'CATS: {n_cats_list}')
    np.random.seed()

    # Reset any existing training files (opening a file in write mode will truncate it)
    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    cat_idx = 0
    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    n_cats = n_cats_list[cat_idx]
                    cat_idx += 1
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
                        # Create different agents to train on
                        agents_to_train_on = []
                        agents_to_train_on.append(UniformSelector(check_assumptions=True, no_baseline=NO_BASELINE))
                        agents_to_train_on.append(FavorMoreRecent(check_assumptions=True, no_baseline=NO_BASELINE))

                        for agent_to_train_on in agents_to_train_on:
                            # Create cats (if any)
                            cats = [AssassinAgent() for _ in range(n_cats)]
                            players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)

                            run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                      numRounds=n_rounds)

    fit_knn_models(enhanced=False)

    # Reset any existing training files (opening a file in write mode will truncate it)
    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    cat_idx = 0
    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    n_cats = n_cats_list[cat_idx]
                    cat_idx += 1
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
                        # Create different agents to train on
                        agents_to_train_on = []
                        agents_to_train_on.append(AlegAATr(lmbda=0.0, ml_model_type='knn', train=True))

                        for agent_to_train_on in agents_to_train_on:
                            # Create cats (if any)
                            cats = [AssassinAgent() for _ in range(n_cats)]
                            players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)

                            run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                      numRounds=n_rounds)

    fit_knn_models(enhanced=True)

    # Train RawR, RRawAAT, RAAT
    NO_BASELINE = True

    # Reset any existing training files (opening a file in write mode will truncate it)
    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    cat_idx = 0
    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    n_cats = n_cats_list[cat_idx]
                    cat_idx += 1
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
                        # Create different agents to train on
                        agents_to_train_on = []
                        agents_to_train_on.append(UniformSelector(check_assumptions=True, no_baseline=NO_BASELINE))
                        agents_to_train_on.append(FavorMoreRecent(check_assumptions=True, no_baseline=NO_BASELINE))

                        for agent_to_train_on in agents_to_train_on:
                            # Create cats (if any)
                            cats = [AssassinAgent() for _ in range(n_cats)]
                            players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)

                            run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                      numRounds=n_rounds)

    train_raw(ENHANCED=False)
    train_qalegaatr(ENHANCED=False)
    train_raat(ENHANCED=False)

    # Reset any existing training files (opening a file in write mode will truncate it)
    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    cat_idx = 0
    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    n_cats = n_cats_list[cat_idx]
                    cat_idx += 1
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
                        # Create different agents to train on
                        agents_to_train_on = []
                        agents_to_train_on.append(RawO(train=True))

                        for agent_to_train_on in agents_to_train_on:
                            # Create cats (if any)
                            cats = [AssassinAgent() for _ in range(n_cats)]
                            players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)

                            run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                      numRounds=n_rounds)

    train_raw(ENHANCED=True)

    # Reset any existing training files (opening a file in write mode will truncate it)
    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    cat_idx = 0
    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    n_cats = n_cats_list[cat_idx]
                    cat_idx += 1
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
                        # Create different agents to train on
                        agents_to_train_on = []
                        agents_to_train_on.append(QAlegAATr(train=True))

                        for agent_to_train_on in agents_to_train_on:
                            # Create cats (if any)
                            cats = [AssassinAgent() for _ in range(n_cats)]
                            players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)

                            run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                      numRounds=n_rounds)

    train_qalegaatr(ENHANCED=True)

    # Reset any existing training files (opening a file in write mode will truncate it)
    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    cat_idx = 0
    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    n_cats = n_cats_list[cat_idx]
                    cat_idx += 1
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
                        # Create different agents to train on
                        agents_to_train_on = []
                        agents_to_train_on.append(RAAT(train=True))

                        for agent_to_train_on in agents_to_train_on:
                            # Create cats (if any)
                            cats = [AssassinAgent() for _ in range(n_cats)]
                            players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)

                            run_with_specified_agents(players, initial_pop_setting=initial_pop_condition,
                                                      numRounds=n_rounds)

    train_raat(ENHANCED=True)

    # Train DQN, RAlegAATr, AleqgAATr
    print('Training EG agents...')
    train_dqn()
    train_ralegaatr()
    train_aleqgaatr()

    # Run the adaptability crap
    print('Generating new adaptability results...')
    adaptability(run_num)
