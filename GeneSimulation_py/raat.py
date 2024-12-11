import csv
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import keras
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from typing import Optional
from utils.utils import BASELINE


@keras.saving.register_keras_serializable()
class AATNetwork(Model):
    def __init__(self, aat_dim: int) -> None:
        super(AATNetwork, self).__init__()
        self.aat_dim = aat_dim

        self.dense1 = Dense(self.aat_dim, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(8, activation='linear')

    def get_config(self):
        return {'aat_dim': self.aat_dim}

    def call(self, aat_state: np.array, return_transformed_state: bool = False) -> tf.Tensor:
        x = self.dense1(aat_state)
        x = self.dense2(x)

        if return_transformed_state:
            return x

        return self.output_layer(x)


class RAAT(AbstractAgent):
    def __init__(self, train: bool = False, enhanced: bool = False, track_vector_file: Optional[str] = None) -> None:
        super().__init__()
        self.whoami = 'RAAT'
        self.generator_pool = GeneratorPool(check_assumptions=True, no_baseline_labels=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        file_adj = '_enh' if enhanced else ''
        self.model = load_model(f'../aat/aat_network/aat_network_model{file_adj}.keras')
        self.aat_scaler = pickle.load(open(f'../aat/aat_network/aat_network_scaler{file_adj}.pickle', 'rb'))
        self.train = train
        self.track_vector_file = track_vector_file
        if self.track_vector_file is not None:
            with open(f'{self.track_vector_file}', 'w', newline='') as _:
                pass
        self.state_dim = (30 ** 2) + (2 * 30)
        self.generators_used = set()

    def _write_to_track_vectors_file(self, vec: np.array) -> None:
        assert self.track_vector_file is not None
        with open(f'{self.track_vector_file}', 'a', newline='') as file:
            writer = csv.writer(file)
            row = np.concatenate([np.array([self.generator_to_use_idx]), vec])
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

        # States
        state_aat = [self.generator_pool.assumptions(generator_idx).alignment_vector() for generator_idx in
                     self.generator_indices]
        state_aat = np.array(state_aat)
        state_aat = state_aat.reshape(1, -1)
        state_aat = self.aat_scaler.transform(state_aat)

        # Make predictions
        q_values = self.model(state_aat)
        self.generator_to_use_idx = np.argmax(q_values.numpy())

        if self.track_vector_file is not None:
            network_state = self.model(state_aat, return_transformed_state=True)
            self._write_to_track_vectors_file(network_state.numpy().reshape(-1, ))

        self.generators_used.add(self.generator_to_use_idx)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        if self.train:
            self.generator_pool.train_aat(player_idx, round_num, received, popularities, influence, extra_data, v,
                                          transactions, self.generator_to_use_idx, BASELINE, enhanced=True)

        # print(f'Generators used: {self.generators_used}')
