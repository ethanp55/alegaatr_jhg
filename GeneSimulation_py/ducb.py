from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np


class DUCB(AbstractAgent):
    def __init__(self, delta: float = 0.99, discount_factor: float = 0.9) -> None:
        super().__init__()
        self.whoami = 'D-UCB'
        self.generator_pool = GeneratorPool()
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.delta = delta
        self.discount_factor = discount_factor
        self.prev_popularity = None
        self.empirical_rewards = {}

        for generator_idx in self.generator_indices:
            self.empirical_rewards[generator_idx] = []

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        curr_popularity = popularities[player_idx]

        # Update empirical results
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.empirical_rewards[self.generator_to_use_idx].append((increase, round_num))
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Pick a generator
        predictions = {}

        for generator_idx in self.generator_indices:
            n_samples = len(self.empirical_rewards[generator_idx])

            if n_samples == 0:
                predictions[generator_idx] = np.inf

            else:
                discounted_rewards, discounted_samples = 0, 0
                for reward, r_num in self.empirical_rewards[generator_idx]:
                    discount = self.discount_factor ** (round_num - r_num)
                    discounted_rewards += discount * reward
                    discounted_samples += discount
                discounted_avg = discounted_rewards / discounted_samples
                upper_bound = ((2 * np.log(1 / self.delta)) / discounted_samples) ** 0.5
                predictions[generator_idx] = discounted_avg + upper_bound

        self.generator_to_use_idx = max(predictions, key=lambda key: predictions[key])

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations
