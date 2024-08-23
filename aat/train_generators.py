from copy import deepcopy
from functools import partial
from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.main import run_with_specified_agents
from GeneSimulation_py.randomagent import RandomAgent
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from typing import List
from utils.utils import BASELINE


# Training labels:
#   - Average popularity increase per round, from the end of the round the generator was used to the end of the game
#   - Baseline = 25 (arbitrary, probably doesn't matter much)

# General training conditions:
#   - 5 different initial popularities (JUST DO EQUAL AND RANDOM FOR NOW)
#   - 5 players, 10 players, 15 players
#   - 20 rounds, 30 rounds, 40 rounds
#   - 0 cats, 1 cat
#   - 30 epochs (TRY 10 FOR NOW)

# Opponents:
#   - CABs with randomly-selected parameters
#   - Random selection of best CABs when trained with no cats
#   - Random selection of best CABs when trained with 1 cat
#   - Random selection of best CABs when trained with 2 cats
#   - Random agents
#   - Basic bandits with epsilon = 0.1, decay = 0.99
#   - Random mixture of all of the above

# Generator training conditions:
#   - Basic bandit with epsilon = 0.1, decay = 0.99 (IGNORE FOR NOW)
#   - Agent that randomly selects generators - uniform
#   - Agent that randomly selects generators - based on how long it's been since last used (more recent is more likely)


# ----------------------------------------------------------------------------------------------------------------------
# HELPER CODE ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Agents used to select generators (for training purposes) -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

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

        return generator_to_token_allocs[self.generator_to_use_idx]


# Agent that just randomly (uniform) chooses a generator to use
class UniformSelector(AbstractAgent):
    def __init__(self, check_assumptions: bool = False) -> None:
        super().__init__()
        self.whoami = 'UniformSelector'
        self.generator_pool = GeneratorPool(check_assumptions=check_assumptions)
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

        return generator_to_token_allocs[self.generator_to_use_idx]


# Agent that favors generators that have been used more recently
class FavorMoreRecent(AbstractAgent):
    def __init__(self, check_assumptions: bool = False) -> None:
        super().__init__()
        self.whoami = 'FavorMoreRecent'
        self.generator_pool = GeneratorPool(check_assumptions=check_assumptions)
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

        return generator_to_token_allocs[self.generator_to_use_idx]


# ----------------------------------------------------------------------------------------------------------------------
# Functions for creating the society of players ------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def cabs_with_random_params(max_players: int = 20) -> List[AbstractAgent]:
    # A CAB will choose random parameters if the gene string is empty ('')
    return [GeneAgent3('', 1) for _ in range(max_players)]


def random_selection_of_best_trained_cabs(folder: str, max_players: int = 20) -> List[AbstractAgent]:
    cabs = []
    files = os.listdir(folder)
    n_generations_to_use = np.random.choice([1, 2, 3])
    n_cabs_per_gen = int(np.ceil(max_players / n_generations_to_use))
    files_to_use = np.random.choice(files, size=n_generations_to_use, replace=False)

    for file in files_to_use:
        df = pd.read_csv(f'{folder}{file}', header=None)

        for i in range(n_cabs_per_gen):
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


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

N_EPOCHS = 5
INITIAL_POP_CONDITIONS = ['equal', 'random']
N_PLAYERS = [5, 10, 15]
N_ROUNDS = [20, 30, 40]
N_CATS = [0, 1]
AUTO_AAT = True


def train_generators() -> None:
    # Variables to track progress
    n_training_iterations = N_EPOCHS * len(INITIAL_POP_CONDITIONS) * len(N_PLAYERS) * len(N_ROUNDS) * len(N_CATS)
    progress_percentage_chunk = int(0.02 * n_training_iterations)
    curr_iteration = 0
    print(n_training_iterations, progress_percentage_chunk)

    # # Reset any existing training files (opening a file in write mode will truncate it)
    # for file in os.listdir('../aat/training_data/'):
    #     with open(f'../aat/training_data/{file}', 'w', newline='') as _:
    #         pass

    # Run the training process
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')

        for initial_pop_condition in INITIAL_POP_CONDITIONS:
            for n_players in N_PLAYERS:
                for n_rounds in N_ROUNDS:
                    for n_cats in N_CATS:
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
                            # Create different agents to train on
                            agents_to_train_on = []
                            # agents_to_train_on.append(BasicBandit(check_assumptions=True))
                            # agents_to_train_on.append(UniformSelector(check_assumptions=True))
                            # agents_to_train_on.append(FavorMoreRecent(check_assumptions=True))
                            agents_to_train_on.append(
                                AlegAATr(lmbda=0.0, ml_model_type='knn', train=True, auto_aat=AUTO_AAT))

                            for agent_to_train_on in agents_to_train_on:
                                # Create cats (if any)
                                cats = [AssassinAgent() for _ in range(n_cats)]
                                players = create_society(agent_to_train_on, cats, deepcopy(opponents), n_players)
                                list_of_players.append(players)

                        # Spin off a process for each grouping of players and play the game
                        pool = Pool(processes=len(list_of_players))
                        pool.map(partial(run_with_specified_agents, initial_pop_setting=initial_pop_condition,
                                         numRounds=n_rounds), list_of_players)

                        curr_iteration += 1


if __name__ == "__main__":
    train_generators()
