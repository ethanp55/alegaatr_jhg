from aat.assumptions import Assumptions
import csv
import fcntl
from GeneSimulation_py.geneagent3 import GeneAgent3
import numpy as np
import pandas as pd
from typing import Dict, Optional


class GeneratorPool:
    def __init__(self, only_use_half: bool = False, check_assumptions: bool = False, auto_aat: bool = False) -> None:
        self.generator_to_assumption_estimates, self.check_assumptions = {}, check_assumptions
        self.auto_aat = auto_aat
        self.generators, generator_df = [], pd.read_csv(f'../ResultsSaved/generator_genes/genes.csv', header=None)

        # Read in the genes for the generators
        step_size = 2 if only_use_half else 1
        for i in range(0, len(generator_df), step_size):
            gene_str = generator_df.iloc[i, 0]
            self.generators.append(GeneAgent3(gene_str, 1, check_assumptions=self.check_assumptions))

        assert len(self.generators) == (8 if only_use_half else 16)

        self.pop_history = []

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array, transactions: np.array,
                   generator_just_used_idx: Optional[int]) -> Dict[int, np.array]:
        self.pop_history.append(popularities)
        generator_to_token_allocs, generator_just_used = {}, None

        if generator_just_used_idx is not None:
            generator_just_used = self.generators[generator_just_used_idx]

            # Grab the assumption estimates, if we're tracking them
            if self.check_assumptions:
                if self.auto_aat:
                    curr_state = np.concatenate(
                        [influence.reshape(-1, 1), popularities.reshape(-1, 1), received.reshape(-1, 1)])
                    n_padding = 300 - curr_state.shape[0]
                    curr_state = np.concatenate([curr_state, np.full((n_padding, 1), -1e9)])
                    tup = (generator_just_used.assumptions(), round_num, np.squeeze(curr_state))

                else:
                    tup = (generator_just_used.assumptions(), round_num, None)
                self.generator_to_assumption_estimates[
                    generator_just_used_idx] = self.generator_to_assumption_estimates.get(generator_just_used_idx,
                                                                                          []) + [tup]

            # Have the generator that was just used run first
            generator_to_token_allocs[generator_just_used_idx] = generator_just_used.play_round(player_idx, round_num,
                                                                                                received, popularities,
                                                                                                influence, extra_data,
                                                                                                v, transactions,
                                                                                                was_just_used=True)

        # For the other generators, set certain assumption parameters to those of the generator that was just used
        for i, generator in enumerate(self.generators):
            if generator_just_used is not None and i == generator_just_used_idx:
                continue

            if generator_just_used is not None:
                generator.detected_comm_just_used = generator_just_used.detected_comm
                generator.desired_comm_just_used = generator_just_used.desired_comm
                generator.prev_attack_tokens_used = generator_just_used.prev_attack_tokens_used
                generator.prev_give_tokens_used = generator_just_used.prev_give_tokens_used
                generator.prev_tokens_kept = generator_just_used.prev_tokens_kept

            generator_to_token_allocs[i] = generator.play_round(player_idx, round_num, received, popularities,
                                                                influence, extra_data, v, transactions)

        return generator_to_token_allocs

    def train_aat(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                  influence: np.array, extra_data, v: np.array, transactions: np.array,
                  generator_just_used_idx: Optional[int], baseline: float,
                  discount_factor: float = 0.9, enhanced: bool = False) -> None:
        if self.auto_aat:
            game_description = 'Repeated, multi-agent, collective-action, general-sum game where players attack or steal in order to build or destroy relationships'
            generator_descriptions = {}
            for generator_idx, generator in enumerate(self.generators):
                generator_descriptions[generator_idx] = ', '.join(f'{k} is {v}' for k, v in generator.genes.items())

        # Calculate assumption estimates for final round
        self.play_round(player_idx, round_num, received, popularities, influence, extra_data, v, transactions,
                        generator_just_used_idx)

        discounted_rewards, running_sum = [0] * (len(self.pop_history) - 1), 0
        for i in reversed(range(len(self.pop_history))):
            if i == 0:
                break
            reward = self.pop_history[i][player_idx] - self.pop_history[i - 1][player_idx]
            running_sum = reward + discount_factor * running_sum
            discounted_rewards[i - 1] = running_sum

        # Store the training data
        for generator_idx, assumptions_history in self.generator_to_assumption_estimates.items():
            for assumption_estimates, round_num, game_state in assumptions_history:
                assert round_num > 0
                assert game_state is not None if self.auto_aat else None

                discounted_reward = discounted_rewards[round_num - 1]
                correction_term = discounted_reward / baseline
                alignment_vector = assumption_estimates.alignment_vector()

                if self.auto_aat:
                    alignment_vector = np.array(alignment_vector).reshape(-1, 1)
                    n_padding = 100 - alignment_vector.shape[0]
                    alignment_vector = np.concatenate([alignment_vector, np.full((n_padding, 1), -1e9)])
                    folder = f'../../auto_aat/train/training_data/jhg/generator_{generator_idx}'

                    with open(f'{folder}_s.csv', 'a', newline='') as file1, \
                            open(f'{folder}_av.csv', 'a', newline='') as file2, \
                            open(f'{folder}_gd.csv', 'a', newline='') as file3, \
                            open(f'{folder}_ed.csv', 'a', newline='') as file4:
                        # Lock the files (for write safety)
                        fcntl.flock(file1.fileno(), fcntl.LOCK_EX)
                        fcntl.flock(file2.fileno(), fcntl.LOCK_EX)
                        fcntl.flock(file3.fileno(), fcntl.LOCK_EX)
                        fcntl.flock(file4.fileno(), fcntl.LOCK_EX)

                        # Write the data
                        writer = csv.writer(file1)
                        writer.writerow(game_state)
                        writer = csv.writer(file2)
                        writer.writerow(np.squeeze(alignment_vector))
                        writer = csv.writer(file3)
                        writer.writerow([generator_descriptions[generator_idx]])
                        writer = csv.writer(file4)
                        writer.writerow([game_description])

                        # Unlock the files
                        fcntl.flock(file1.fileno(), fcntl.LOCK_UN)
                        fcntl.flock(file2.fileno(), fcntl.LOCK_UN)
                        fcntl.flock(file3.fileno(), fcntl.LOCK_UN)
                        fcntl.flock(file4.fileno(), fcntl.LOCK_UN)

                else:
                    # Store the alignment vector
                    adjustment = '_enh' if enhanced else ''
                    file_path = f'../aat/training_data/generator_{generator_idx}_vectors{adjustment}.csv'

                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(alignment_vector)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the correction term
                    file_path = f'../aat/training_data/generator_{generator_idx}_correction_terms{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow([correction_term])
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

    def assumptions(self, generator_idx: int) -> Assumptions:
        return self.generators[generator_idx].assumptions()
