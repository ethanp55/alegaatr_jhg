from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np
import random


class EXP4(AbstractAgent):
    def __init__(self, gamma: float = 0.5) -> None:
        super().__init__()
        self.whoami = 'EXP4'
        self.generator_pool = GeneratorPool()
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.gamma, self.k = gamma, len(self.generator_indices)
        self.prev_popularity = None
        self.weights, self.p_t_vals = {}, None

        for generator_idx in self.generator_indices:
            self.weights[generator_idx] = 1.0

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        curr_popularity = popularities[player_idx]

        # Update empirical results
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            x_hat = increase / self.p_t_vals[self.generator_to_use_idx]
            self.weights[self.generator_to_use_idx] = self.weights[self.generator_to_use_idx] * np.exp(
                (self.gamma * x_hat) / self.k)
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Pick a generator
        weight_sum = sum(self.weights.values())
        self.p_t_vals = [((1 - self.gamma) * (weight / weight_sum)) + (self.gamma / self.k) for weight in
                         self.weights.values()]
        self.generator_to_use_idx = random.choices(list(self.weights.keys()), weights=self.p_t_vals, k=1)[0]

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations
