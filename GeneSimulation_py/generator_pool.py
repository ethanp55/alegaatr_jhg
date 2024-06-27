from aat.assumptions import Assumptions
import csv
import fcntl
from GeneSimulation_py.geneagent3 import GeneAgent3
import numpy as np
import pandas as pd
from typing import Dict, Optional


class GeneratorPool:
    def __init__(self, only_use_half: bool = False, check_assumptions: bool = False) -> None:
        self.generator_to_assumption_estimates, self.check_assumptions = {}, check_assumptions
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
                self.generator_to_assumption_estimates[
                    generator_just_used_idx] = self.generator_to_assumption_estimates.get(generator_just_used_idx,
                                                                                          []) + [
                                                   (generator_just_used.assumptions(), round_num)]

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
                  generator_just_used_idx: Optional[int], baseline_increase: float) -> None:
        # Calculate assumption estimates for final round
        self.play_round(player_idx, round_num, received, popularities, influence, extra_data, v, transactions,
                        generator_just_used_idx)

        # Store the training data
        for generator_idx, assumptions_history in self.generator_to_assumption_estimates.items():
            for assumption_estimates, round_num in assumptions_history:
                assert round_num > 0

                avg_increase = (self.pop_history[-1][player_idx] - self.pop_history[round_num - 1][player_idx]) / (
                        len(self.pop_history) - round_num)
                correction_term = avg_increase / baseline_increase
                alignment_vector = assumption_estimates.alignment_vector()

                # Store the alignment vector
                file_path = f'../aat/training_data/generator_{generator_idx}_vectors.csv'
                with open(file_path, 'a', newline='') as file:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                    writer = csv.writer(file)
                    writer.writerow(alignment_vector)
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                # Store the correction term
                file_path = f'../aat/training_data/generator_{generator_idx}_correction_terms.csv'
                with open(file_path, 'a', newline='') as file:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                    writer = csv.writer(file)
                    writer.writerow([correction_term])
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

    def assumptions(self, generator_idx: int) -> Assumptions:
        return self.generators[generator_idx].assumptions()
