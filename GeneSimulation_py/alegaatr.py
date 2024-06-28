from aat.train_generators import BASELINE
from collections import deque
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np
import os
import pickle


class AlegAATr(AbstractAgent):
    def __init__(self, lmbda: float = 0.95, ml_model_type: str = 'knn', lookback: int = 5) -> None:
        super().__init__()
        self.whoami = 'AlegAATr'
        self.lmbda = lmbda
        self.generator_pool = GeneratorPool(check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.models, self.scalers = {}, {}
        self._read_in_generator_models(ml_model_type)
        self.empirical_increases, self.n_rounds_since_used = {}, {}
        self._initialize_empirical_data(lookback)
        self.prev_popularity = None

    def _read_in_generator_models(self, ml_model_type: str) -> None:
        folder = '../aat/knn_models/' if ml_model_type == 'knn' else '../aat/nn_models/'

        for file in os.listdir(folder):
            generator_idx = int(file.split('_')[1])
            full_file_path = f'{folder}{file}'

            if 'scaler' in file:
                self.scalers[generator_idx] = pickle.load(open(full_file_path, 'rb'))

            else:
                self.models[generator_idx] = pickle.load(open(full_file_path, 'rb'))

    def _initialize_empirical_data(self, lookback: int) -> None:
        for generator_idx in self.generator_indices:
            self.empirical_increases[generator_idx] = deque(maxlen=lookback)
            self.n_rounds_since_used[generator_idx] = 1

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        curr_popularity = popularities[player_idx]

        # Update empirical results
        if self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.empirical_increases[self.generator_to_use_idx].append(increase)
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Make predictions for each generator
        best_pred, best_generator_idx = -np.inf, None

        for generator_idx in self.generator_indices:
            n_rounds_since_last_use = self.n_rounds_since_used[generator_idx]

            # Use empirical results as the prediction
            if np.random.rand() < self.lmbda ** n_rounds_since_last_use and len(
                    self.empirical_increases[generator_idx]) > 0:
                increases = self.empirical_increases[generator_idx]
                avg = sum(increases) / len(increases)
                pred = avg

            # Otherwise, use AAT
            else:
                generator_assumption_estimates = self.generator_pool.assumptions(generator_idx)
                x = np.array(generator_assumption_estimates.alignment_vector()).reshape(1, -1)
                x_scaled = self.scalers[generator_idx].transform(x) if generator_idx in self.scalers else x
                correction_term_pred = self.models[generator_idx].predict(x_scaled)[0]
                pred = BASELINE * correction_term_pred

            if pred > best_pred:
                best_pred, best_generator_idx = pred, generator_idx

        self.generator_to_use_idx = best_generator_idx

        # Update how many rounds it has been since each generator has been used
        for generator_idx in self.n_rounds_since_used.keys():
            if generator_idx == self.generator_to_use_idx:
                self.n_rounds_since_used[generator_idx] = 1

            else:
                self.n_rounds_since_used[generator_idx] += 1

        return generator_to_token_allocs[self.generator_to_use_idx]
