from collections import deque
from copy import deepcopy
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.generator_pool import GeneratorPool
import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow import keras


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


class MADQN(AbstractAgent):
    def __init__(self, max_n_players: int = 30, learning_rate: float = 0.001, discount_factor: float = 0.9,
                 epsilon: float = 0.1, epsilon_decay: float = 0.99, replay_buffer_size: int = 5000,
                 batch_size: int = 128, train_networks: bool = False) -> None:
        super().__init__()
        self.whoami = 'MADQN'

        # Generators
        self.generator_pool = GeneratorPool()
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

        # Variables used for training and/or action selection
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.state = None
        self.train_networks = train_networks
        self.prev_popularity = None
        self.best_losses = [np.inf for _ in self.generator_indices]

        # DQN and target models
        self.state_dim = (max_n_players ** 2) + (
                2 * max_n_players)  # Flatten the influence matrix and add the popularity and received matrices
        self.action_dim = 1
        self.models = [DQN(self.state_dim, self.action_dim) for _ in self.generator_indices]
        self.target_models = [DQN(self.state_dim, self.action_dim) for _ in self.generator_indices]
        for i in self.generator_indices:
            model, target_model = self.models[i], self.target_models[i]
            target_model.set_weights(model.get_weights())

        # Optimizers
        self.optimizers = [tf.optimizers.Adam(learning_rate=learning_rate) for _ in self.generator_indices]

        # Replay buffer
        self.replay_buffers = [deque(maxlen=replay_buffer_size) for _ in self.generator_indices]

        # Episode experiences (to add to the replay buffers)
        self.current_episode_experiences = [[] for _ in self.generator_indices]

        # State scaler
        self.scalers = [DynamicMinMaxScaler(self.state_dim) for _ in self.generator_indices]

        # If we're not in training mode, load the saved/trained models
        if not self.train_networks:
            self.load_networks()

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
        if self.train_networks:
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
        if self.train_networks and self.prev_popularity is not None:
            increase = curr_popularity - self.prev_popularity
            self.add_experience(self.generator_to_use_idx, increase, next_state, False)
        self.prev_popularity = curr_popularity

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.play_round(player_idx, round_num, received, popularities,
                                                                   influence, extra_data, v, transactions,
                                                                   self.generator_to_use_idx)

        self.state = next_state

        # Epsilon-greedy policy for generator selection (only consider exploring if we're in training mode)
        if self.train_networks and np.random.rand() < self.epsilon:
            self.generator_to_use_idx = np.random.choice(self.generator_indices)

        else:
            q_vals = []

            for i in self.generator_indices:
                scaled_state = self.scalers[i].scale(self.state)
                q_values = self.models[i](np.expand_dims(scaled_state, 0))
                q_vals.append(q_values.numpy())

            self.generator_to_use_idx = np.argmax(q_vals)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]
        self.generator_pool.update_generator_allocations(token_allocations)

        return token_allocations

    def train(self) -> None:
        for i in self.generator_indices:
            if len(self.replay_buffers[i]) < self.batch_size:
                continue

            model, target_model, optimizer = self.models[i], self.target_models[i], self.optimizers[i]

            for _ in range(100):
                best_loss = self.best_losses[i]

                # Sample a batch of experiences from the replay buffer
                batch = random.sample(self.replay_buffers[i], self.batch_size)
                batch_states, batch_rewards, batch_next_states, batch_dones = map(np.array, zip(*batch))

                # Q-learning update using the DQN loss
                next_q_values = target_model(batch_next_states)
                targets = batch_rewards + (1 - batch_dones) * self.discount_factor * tf.squeeze(next_q_values)

                with tf.GradientTape() as tape:
                    q_values = model(batch_states)
                    loss = tf.keras.losses.MSE(targets, tf.squeeze(q_values))

                loss_val = loss.numpy()
                if loss_val < best_loss:
                    print(f'Loss {i} improved from {best_loss} to {loss_val}')
                    self.best_losses[i] = loss_val
                    self.save_networks()

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            target_model.set_weights(model.get_weights())

    def add_experience(self, action: int, reward: float, next_state: np.array, done: bool):
        # Accumulate experiences over multiple time steps
        scaled_state = self.scalers[action].scale(self.state)
        scaled_next_state = self.scalers[action].scale(next_state)
        self.scalers[action].update(next_state)

        self.current_episode_experiences[action].append((scaled_state, reward, scaled_next_state, done))

        # If the episode is done, add the accumulated experiences to the replay buffers
        if done:
            for i in self.generator_indices:
                self.replay_buffers[i].extend(self.current_episode_experiences[i])
                self.current_episode_experiences[i] = []

    def clear_buffer(self) -> None:
        for i in self.generator_indices:
            self.replay_buffers[i].clear()
            self.current_episode_experiences[i] = []

    def save_networks(self) -> None:
        # Save the networks and scalers
        for i in range(len(self.generator_indices)):
            self.models[i].save(f'../GeneSimulation_py/madqn_model/model_{i}.keras')

            with open(f'../GeneSimulation_py/madqn_model/scaler_{i}.pickle', 'wb') as f:
                pickle.dump(self.scalers[i], f)

    def load_networks(self) -> None:
        # Load the networks and scalers
        self.models, self.scalers = [], []

        for i in range(len(self.generator_indices)):
            self.models.append(keras.models.load_model(f'../GeneSimulation_py/madqn_model/model_{i}.keras'))
            self.scalers.append(pickle.load(open(f'../GeneSimulation_py/madqn_model/scaler_{i}.pickle', 'rb')))
