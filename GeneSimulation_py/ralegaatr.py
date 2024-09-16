from collections import deque
from copy import deepcopy
import csv
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from typing import Optional


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


class RAlegAATr(AbstractAgent):
    def __init__(self, learning_rate: float = 0.001, discount_factor: float = 0.9, epsilon: float = 0.1,
                 epsilon_decay: float = 0.99, replay_buffer_size: int = 5000, batch_size: int = 256,
                 train_network: bool = False, track_vector_file: Optional[str] = None) -> None:
        super().__init__()
        self.whoami = 'RAlegAATr'

        # Variables used for training and/or action selection
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.state = None
        self.training_started = False
        self.train_network = train_network
        self.prev_popularity = None
        self.best_loss = np.inf

        # Generators
        self.generator_pool = GeneratorPool(check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

        # Model and target model
        self.state_dim = len(self.generator_indices) * 59
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
        self.generator_pool = GeneratorPool(check_assumptions=True)
        self.generator_to_use_idx = None
        self.prev_popularity = None

    def record_final_results(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                             influence: np.array, extra_data, v: np.array, transactions: np.array) -> None:
        # Add a new experience to the replay buffer and update the network weights (if there are enough experiences)
        if self.train_network:
            curr_popularity = popularities[player_idx]
            next_state = [self.generator_pool.assumptions(generator_idx).alignment_vector() for generator_idx in
                          self.generator_indices]
            next_state = np.array(next_state)
            next_state = next_state.reshape(1, -1)
            increase = curr_popularity - self.prev_popularity
            self.add_experience(self.generator_to_use_idx, increase, next_state, True)
            self.train()

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array) -> np.array:
        # Add a new experience to the replay buffer and update the network weights (if there are enough experiences)
        curr_popularity = popularities[player_idx]
        next_state = [self.generator_pool.assumptions(generator_idx).alignment_vector() for generator_idx in
                      self.generator_indices]
        next_state = np.array(next_state)
        next_state = next_state.reshape(1, -1)
        if self.train_network and self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.add_experience(self.generator_to_use_idx, increase, next_state, False)
            self.train()
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
            q_values = self.model(self.state)
            self.generator_to_use_idx = np.argmax(q_values.numpy())

            if self.track_vector_file is not None:
                network_state = self.model(self.state, return_transformed_state=True)
                self._write_to_track_vectors_file(network_state.numpy().reshape(-1, ))

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations

    def update_networks(self) -> None:
        # Update target network weights periodically
        if self.training_started:
            self.target_model.set_weights(self.model.get_weights())

    def train(self) -> None:
        if len(self.replay_buffer) < self.replay_buffer.maxlen:
            return

        self.training_started = True

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = map(np.array, zip(*batch))
        batch_states, batch_next_states = \
            batch_states.reshape(self.batch_size, -1), batch_next_states.reshape(self.batch_size, -1)

        # Q-value update
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
        self.current_episode_experiences.append((self.state, action, reward, next_state, done))

        # If the episode is done, add the accumulated experiences to the replay buffer
        if done:
            self.replay_buffer.extend(self.current_episode_experiences)
            self.current_episode_experiences = []

    def save_network(self) -> None:
        # Save the network
        self.model.save('../GeneSimulation_py/ralegaatr_model/model.keras')

    def load_network(self) -> None:
        # Load the network
        self.model = keras.models.load_model('../GeneSimulation_py/ralegaatr_model/model.keras')
