import csv
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import load_model, Model
from typing import Optional, Tuple
from utils.utils import BASELINE


@keras.saving.register_keras_serializable()
class AutoBaseline(Model):
    def __init__(self, aat_dim: int, action_dim: int) -> None:
        super(AutoBaseline, self).__init__()
        self.aat_dim = aat_dim
        self.action_dim = action_dim

        self.embedding_layer_1 = Embedding(input_dim=self.action_dim, output_dim=32)
        self.dense_embedding_1 = Dense(self.aat_dim, activation='relu')

        self.embedding_layer_2 = Embedding(input_dim=self.action_dim, output_dim=32)
        self.dense_embedding_2 = Dense(self.aat_dim, activation='relu')

        self.dense_aat_1 = Dense(32, activation='relu')
        self.dense_aat_2 = Dense(self.action_dim, activation='sigmoid')

        self.dense_val_1 = Dense(32, activation='relu')
        self.dense_val_2 = Dense(self.action_dim, activation='linear')

        self.dense_correction_1 = Dense(self.action_dim, activation='relu')
        self.dense_correction_2 = Dense(self.action_dim, activation='linear')

        self.output_layer = Dense(1, activation='linear')

    def get_config(self):
        return {'aat_dim': self.aat_dim, 'action_dim': self.action_dim}

    def call(self, input: Tuple[np.array, int]) -> tf.Tensor:
        aat_state, generator_idx = input

        generator_embedding_aat = self.embedding_layer_1(generator_idx)
        generator_embedding_aat = self.dense_embedding_1(generator_embedding_aat)

        generator_embedding_val_pred = self.embedding_layer_2(generator_idx)
        generator_embedding_val_pred = self.dense_embedding_2(generator_embedding_val_pred)

        aat_compressed = self.dense_aat_1(aat_state + generator_embedding_aat)
        aat_compressed = self.dense_aat_2(aat_compressed)

        val_pred = self.dense_val_1(aat_state + generator_embedding_val_pred)
        val_pred = self.dense_val_2(val_pred)

        combined = aat_compressed + val_pred

        correction = self.dense_correction_1(combined)
        correction = self.dense_correction_2(correction)

        return self.output_layer(val_pred * correction)


class AAlegAATr(AbstractAgent):
    def __init__(self, train: bool = False, enhanced: bool = False, generator_usage_file: Optional[str] = None,
                 track_vector_file: Optional[str] = None) -> None:
        super().__init__()
        self.whoami = 'AAlegAATr'
        self.generator_pool = GeneratorPool(check_assumptions=True, no_baseline_labels=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        file_adj = '_enh' if enhanced else ''
        self.model = load_model(f'../aat/auto_baseline_model/auto_baseline{file_adj}.keras')
        self.train = train
        self.generator_usage_file = generator_usage_file
        if self.generator_usage_file is not None:
            with open(f'{self.generator_usage_file}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['round', 'generator'])
        self.track_vector_file = track_vector_file
        if self.track_vector_file is not None:
            with open(f'{self.track_vector_file}.csv', 'w', newline='') as _:
                pass
        self.generators_used = set()

    def _write_to_generator_usage_file(self, round_num: int) -> None:
        assert self.generator_usage_file is not None
        with open(f'{self.generator_usage_file}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round_num, self.generator_to_use_idx])

    def _write_to_track_vectors_file(self, alignment_vector: np.array) -> None:
        assert self.track_vector_file is not None
        with open(f'{self.track_vector_file}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            row = np.concatenate([np.array([self.generator_to_use_idx]), alignment_vector[0, :]])
            writer.writerow(np.squeeze(row))

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        # Make predictions for each generator
        best_pred, best_generator_idx, best_vector = -np.inf, None, None

        for generator_idx in self.generator_indices:
            generator_assumption_estimates = self.generator_pool.assumptions(generator_idx)
            x = np.array(generator_assumption_estimates.alignment_vector()).reshape(1, -1)
            pred = self.model((x, np.array([generator_idx]))).numpy()[0][0]

            if pred > best_pred:
                best_pred, best_generator_idx = pred, generator_idx
                best_vector = x

        prev_generator_idx = self.generator_to_use_idx
        self.generator_to_use_idx = best_generator_idx

        if self.generator_to_use_idx != prev_generator_idx and self.generator_usage_file is not None:
            self._write_to_generator_usage_file(round_num)

        if self.track_vector_file is not None and best_vector is not None:
            self._write_to_track_vectors_file(best_vector)

        self.generators_used.add(self.generator_to_use_idx)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        if self.train:
            self.generator_pool.train_aat(player_idx, round_num, received, popularities, influence, extra_data, v,
                                          transactions, self.generator_to_use_idx, BASELINE, enhanced=True)

        print(f'Generators used: {self.generators_used}')