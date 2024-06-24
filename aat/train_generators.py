from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.randomagent import RandomAgent
import numpy as np
import os
import pandas as pd
from typing import List


# TODO: ask Dr. Crandall about both of these points
# How to create training labels:
#   - Average popularity increase per round, from the end of the current round to the end of the game
#   - Baseline = 25

# General training conditions:
#   - 5 different initial popularities
#   - 5 players, 10 players, 15 players, 20 players
#   - 20 rounds, 30 rounds, 40 rounds, 50 rounds, 100 rounds
#   - 0 cats, 1 cat, 2 cats
#   - Multiple epochs (maybe 30?)

# Opponents:
#   - Random selection of generators
#   - CABs with randomly-selected parameters
#   - Random selection of best CABs when trained with no cats
#   - Random selection of best CABs when trained with 1 cat
#   - Random selection of best CABs when trained with 2 cats
#   - Random agents
#   - Random mixture of all of the above

# Generator training conditions:
#   - Basic bandit with epsilon = 0.05, decay = 0.99
#   - Basic bandit with epsilon = 0.1, decay = 0.99
#   - Basic bandit with epsilon = 0.2, decay = 0.99
#   - Agent that randomly selects generators - uniform
#   - Agent that randomly selects generators - based on how long it's been since last used (more recent is more likely)
#   - BBL
#   - S++

# ----------------------------------------------------------------------------------------------------------------------
# Functions for creating the society of players ------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def random_selection_of_generators(max_players: int = 20) -> List[AbstractAgent]:
    generators, generator_df = [], pd.read_csv(f'../ResultsSaved/generator_genes/genes.csv', header=None)

    # Read in the genes for the generators
    for i in range(len(generator_df)):
        gene_str = generator_df.iloc[i, 0]
        generators.append(GeneAgent3(gene_str, 1))

    return list(np.random.choice(generators, size=max_players, replace=True))


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


def random_mixture_of_all_types(max_players: int = 20) -> List[AbstractAgent]:
    agents = []
    agents.extend(random_selection_of_generators(max_players))
    agents.extend(cabs_with_random_params(max_players))
    agents.extend(random_selection_of_best_trained_cabs('../ResultsSaved/no_cat/', max_players))
    agents.extend(random_selection_of_best_trained_cabs('../ResultsSaved/one_cat/', max_players))
    agents.extend(random_selection_of_best_trained_cabs('../ResultsSaved/two_cats/', max_players))
    agents.extend(random_agents(max_players))

    np.random.shuffle(agents)

    return agents[:max_players]


def create_society(our_player: AbstractAgent, all_other_players: List[AbstractAgent],
                   n_players: int) -> List[AbstractAgent]:
    players = []
    np.random.shuffle(all_other_players)

    for i in range(n_players - 1):
        players.append(all_other_players[i])

    players.append(our_player)

    return players


# ----------------------------------------------------------------------------------------------------------------------
# Agents used to select generators (for training purposes) -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Simple bandit agent that periodically explores and exploits otherwise
class BasicBandit(AbstractAgent):
    def __init__(self, epsilon: float, epsilon_decay: float, use_half_of_generators: bool = False) -> None:
        super().__init__()
        self.whoami = 'BasicBandit'
        self.epsilon, self.epsilon_decay = epsilon, epsilon_decay
        self.generator_pool = GeneratorPool(only_use_half=use_half_of_generators, check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.empirical_increases = {}
        self.player_idx, self.prev_popularity = None, None

    def setGameParams(self, game_params, visual_traits, forced_random) -> None:
        pass

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array) -> np.array:
        if self.player_idx is None:
            self.player_idx = player_idx

        # Update empirical rewards
        curr_popularity = popularities[self.player_idx]
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.empirical_increases[self.generator_to_use_idx] = self.empirical_increases.get(self.empirical_increases,
                                                                                               []) + [increase]
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, self.generator_to_use_idx)
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
    def __init__(self, use_half_of_generators: bool = False) -> None:
        super().__init__()
        self.whoami = 'UniformSelector'
        self.generator_pool = GeneratorPool(only_use_half=use_half_of_generators, check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

    def setGameParams(self, game_params, visual_traits, forced_random) -> None:
        pass

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array) -> np.array:
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, self.generator_to_use_idx)

        # Randomly (uniform) choose a generator to use
        self.generator_to_use_idx = np.random.choice(self.generator_indices)

        return generator_to_token_allocs[self.generator_to_use_idx]


class FavorMoreRecent(AbstractAgent):
    def __init__(self, use_half_of_generators: bool = False) -> None:
        super().__init__()
        self.whoami = 'FavorMoreRecent'
        self.generator_pool = GeneratorPool(only_use_half=use_half_of_generators, check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx, self.prev_generator_idx = None, None
        self.n_rounds_since_last_use = {}
        self.max_in_a_row = 5
        self.n_rounds_used = 0

    def setGameParams(self, game_params, visual_traits, forced_random) -> None:
        pass

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array) -> np.array:
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, self.generator_to_use_idx)

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
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


BASELINE = 25
INITIAL_POP_CONDITIONS = ['equal', 'highlow', 'power', 'random', 'step']
N_PLAYERS = [5, 10, 15, 20]
N_ROUNDS = [20, 30, 40, 50, 100]
N_CATS = [0, 1, 2]
N_EPOCHS = 30
