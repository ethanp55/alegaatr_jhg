from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np


class RUCB(AbstractAgent):
    def __init__(self, delta: float = 0.99, reset_every_n_rounds: int = 5) -> None:
        super().__init__()
        self.whoami = 'R-UCB'
        self.generator_pool = GeneratorPool()
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.delta = delta
        self.reset_every_n_rounds = reset_every_n_rounds
        self.prev_popularity = None
        self.empirical_rewards, self.n_samples = {}, {}

        for generator_idx in self.generator_indices:
            self.empirical_rewards[generator_idx] = 0
            self.n_samples[generator_idx] = 0

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def _reset(self) -> None:
        for generator_idx in self.generator_indices:
            self.empirical_rewards[generator_idx] = 0
            self.n_samples[generator_idx] = 0

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        # Reset the empirical rewards and samples data every self.reset_every_n_rounds rounds
        if round_num % self.reset_every_n_rounds == 0:
            self._reset()

        curr_popularity = popularities[player_idx]

        # Update empirical results
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.empirical_rewards[self.generator_to_use_idx] += increase
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Pick a generator
        predictions = {}

        for generator_idx in self.generator_indices:
            n_samples = self.n_samples[generator_idx]

            if n_samples == 0:
                predictions[generator_idx] = np.inf

            else:
                empirical_avg = self.empirical_rewards[generator_idx] / n_samples
                upper_bound = ((2 * np.log(1 / self.delta)) / n_samples) ** 0.5
                predictions[generator_idx] = empirical_avg + upper_bound

        self.generator_to_use_idx = max(predictions, key=lambda key: predictions[key])
        self.n_samples[self.generator_to_use_idx] += 1

        return generator_to_token_allocs[self.generator_to_use_idx]
