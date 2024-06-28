from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np
import random


class EEE(AbstractAgent):
    def __init__(self, explore_prob: float = 0.1) -> None:
        super().__init__()
        self.whoami = 'EEE'
        self.generator_pool = GeneratorPool()
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.explore_prob = explore_prob
        self.prev_popularity = None
        self.m_e, self.n_e, self.s_e = {}, {}, {}
        self.in_phase, self.phase_counter, self.phase_rewards, self.n_i = False, 0, [], 0

        for generator_idx in self.generator_indices:
            self.m_e[generator_idx] = 0
            self.n_e[generator_idx] = 0
            self.s_e[generator_idx] = 0

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        curr_popularity = popularities[player_idx]

        # Update empirical results
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.phase_rewards.append(increase)
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Pick a generator
        if self.in_phase:
            if self.phase_counter < self.n_i:
                self.phase_counter += 1

            else:
                avg_phase_reward = np.array(self.phase_rewards).mean() if len(self.phase_rewards) > 0 else 0
                self.n_e[self.generator_to_use_idx] += 1
                self.s_e[self.generator_to_use_idx] += self.n_i
                self.m_e[self.generator_to_use_idx] = self.m_e[self.generator_to_use_idx] + (
                            self.n_i / self.s_e[self.generator_to_use_idx]) * (avg_phase_reward - self.m_e[
                    self.generator_to_use_idx])
                self.phase_rewards, self.phase_counter, self.n_i, self.in_phase = [], 0, 0, False

        if not self.in_phase:
            explore = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])

            if explore:
                new_agent = random.choice(self.generator_indices)

                self.generator_to_use_idx = new_agent

            else:
                max_reward, agents_to_consider = max(list(self.m_e.values())), []

                for key, val in self.m_e.items():
                    if val == max_reward:
                        agents_to_consider.append(key)

                new_agent = random.choice(agents_to_consider)

                self.generator_to_use_idx = new_agent

            self.n_i, self.in_phase = np.random.choice(list(range(1, 5))), True

        return generator_to_token_allocs[self.generator_to_use_idx]
