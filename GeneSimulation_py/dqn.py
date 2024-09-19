from collections import deque
from copy import deepcopy
import csv
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow import keras
from typing import Optional


class DynamicMinMaxScaler:
    def __init__(self, num_features: int) -> None:
        self.num_features = num_features
        self.min_vals = np.inf * np.ones(num_features)
        self.max_vals = -np.inf * np.ones(num_features)

    def update(self, state: np.array) -> None:
        self.min_vals = np.minimum(self.min_vals, state)
        self.max_vals = np.maximum(self.max_vals, state)

    def scale(self, state: np.array) -> np.array:
        return (state - self.min_vals) / (self.max_vals - self.min_vals + 0.00001)


class DQN(keras.Model):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dense1 = keras.layers.Dense(self.state_dim, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.output_layer = keras.layers.Dense(self.action_dim, activation='linear')

    def get_config(self):
        return {'state_dim': self.state_dim, 'action_dim': self.action_dim}

    def call(self, state: np.array, return_transformed_state: bool = False) -> tf.Tensor:
        x = self.dense1(state)
        x = self.dense2(x)

        if return_transformed_state:
            return x

        return self.output_layer(x)


class DQNAgent(AbstractAgent):
    def __init__(self, max_n_players: int = 30, learning_rate: float = 0.001, discount_factor: float = 0.9,
                 epsilon: float = 0.1, epsilon_decay: float = 0.99, replay_buffer_size: int = 50000,
                 batch_size: int = 256, train_network: bool = False, track_vector_file: Optional[str] = None) -> None:
        super().__init__()
        self.whoami = 'DQN'

        # Variables used for training and/or action selection
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.state = None
        self.train_network = train_network
        self.prev_popularity = None
        self.best_loss = np.inf

        # Generators
        self.generator_pool = GeneratorPool()
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

        # DQN model and target model
        self.state_dim = (max_n_players ** 2) + (
                2 * max_n_players)  # Flatten the influence matrix and add the popularity and received matrices
        self.action_dim = len(self.generator_indices)
        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.target_model.set_weights(self.model.get_weights())

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Episode experiences (to add to the replay buffer)
        self.current_episode_experiences = []

        # State scaler
        self.scaler = DynamicMinMaxScaler(self.state_dim)

        # If we're not in training mode, load the saved/trained model
        if not self.train_network:
            self.load_network()

        self.track_vector_file = track_vector_file
        if self.track_vector_file is not None:
            with open(f'{self.track_vector_file}.csv', 'w', newline='') as _:
                pass

    def _write_to_track_vectors_file(self, state_vector: np.array) -> None:
        assert self.track_vector_file is not None
        with open(f'{self.track_vector_file}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            row = np.concatenate([np.array([self.generator_to_use_idx]), state_vector])
            writer.writerow(np.squeeze(row))

    def setGameParams(self, game_params, forced_random) -> None:
        for generator in self.generator_pool.generators:
            generator.setGameParams(game_params, forced_random)

    def update_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay

    def reset(self) -> None:
        self.generator_pool = GeneratorPool()
        self.generator_to_use_idx = None
        self.prev_popularity = None

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        # Add a new experience to the replay buffer and update the network weights (if there are enough experiences)
        if self.train_network:
            curr_popularity = popularities[player_idx]
            next_state = deepcopy(influence)
            next_state = np.concatenate(
                [next_state.reshape(-1, 1), popularities.reshape(-1, 1), received.reshape(-1, 1)])
            n_zeroes_for_state = self.state_dim - next_state.shape[0]
            next_state = np.append(next_state, np.zeros(n_zeroes_for_state))
            increase = curr_popularity - self.prev_popularity
            self.add_experience(self.generator_to_use_idx, increase, next_state, True)

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        # Add a new experience to the replay buffer and update the network weights (if there are enough experiences)
        curr_popularity = popularities[player_idx]
        next_state = deepcopy(influence)
        next_state = np.concatenate([next_state.reshape(-1, 1), popularities.reshape(-1, 1), received.reshape(-1, 1)])
        n_zeroes_for_state = self.state_dim - next_state.shape[0]
        next_state = np.append(next_state, np.zeros(n_zeroes_for_state))
        if self.train_network and self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.add_experience(self.generator_to_use_idx, increase, next_state, False)
        if self.prev_popularity is None:
            self.scaler.update(next_state)
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        self.state = next_state

        # Epsilon-greedy policy for generator selection (only consider exploring if we're in training mode)
        if self.train_network and np.random.rand() < self.epsilon:
            self.generator_to_use_idx = np.random.choice(self.action_dim)

        else:
            scaled_state = self.scaler.scale(self.state)
            q_values = self.model(np.expand_dims(scaled_state, 0))
            self.generator_to_use_idx = np.argmax(q_values.numpy())

            if self.track_vector_file is not None:
                network_state = self.model(np.expand_dims(scaled_state, 0), return_transformed_state=True)
                self._write_to_track_vectors_file(network_state.numpy().reshape(-1, ))

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations

    def update_networks(self) -> None:
        # Update target network weights periodically
        self.target_model.set_weights(self.model.get_weights())

    def train(self) -> None:
        for _ in range(100):
            # Sample a batch of experiences from the replay buffer
            batch = random.sample(self.replay_buffer, self.batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = map(np.array, zip(*batch))

            # Q-learning update using the DQN loss
            next_q_values = self.target_model(batch_next_states)
            max_next_q_values = np.max(next_q_values.numpy(), axis=1)

            targets = batch_rewards + (1 - batch_dones) * self.discount_factor * max_next_q_values

            with tf.GradientTape() as tape:
                q_values = self.model(batch_states)
                selected_action_values = tf.reduce_sum(tf.one_hot(batch_actions, self.action_dim) * q_values, axis=1)
                loss = tf.keras.losses.MSE(targets, selected_action_values)

            loss_val = loss.numpy()
            if loss_val < self.best_loss:
                print(f'Loss improved from {self.best_loss} to {loss_val}')
                self.best_loss = loss_val
                self.save_network()

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def add_experience(self, action: int, reward: float, next_state: np.array, done: bool):
        # Accumulate experiences over multiple time steps
        scaled_state = self.scaler.scale(self.state)
        scaled_next_state = self.scaler.scale(next_state)
        self.scaler.update(next_state)

        self.current_episode_experiences.append((scaled_state, action, reward, scaled_next_state, done))

        # If the episode is done, add the accumulated experiences to the replay buffer
        if done:
            self.replay_buffer.extend(self.current_episode_experiences)
            self.current_episode_experiences = []

    def clear_buffer(self) -> None:
        self.replay_buffer.clear()
        self.current_episode_experiences = []

    def save_network(self) -> None:
        # Save the network and scaler
        self.model.save('../GeneSimulation_py/dqn_model/model.keras')

        with open('../GeneSimulation_py/dqn_model/scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_network(self) -> None:
        # Load the network and scaler
        self.model = keras.models.load_model('../GeneSimulation_py/dqn_model/model.keras')
        self.scaler = pickle.load(open('../GeneSimulation_py/dqn_model/scaler.pickle', 'rb'))
