import csv
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

    def play_round(self, player_idx: int, round_num: int, received: np.array, popularities: np.array,
                   influence: np.array, extra_data, v: np.array,
                   generator_just_used_idx: Optional[int]) -> Dict[int, np.array]:
        generator_to_token_allocs, generator_just_used = {}, None

        # Have the generator that was just used run first
        if generator_just_used_idx is not None:
            generator_just_used = self.generators[generator_just_used_idx]
            generator_to_token_allocs[generator_just_used_idx] = generator_just_used.play_round(player_idx, round_num,
                                                                                                received, popularities,
                                                                                                influence, extra_data,
                                                                                                v, was_just_used=True)

            # Grab the assumption estimates, if we're tracking them
            if self.check_assumptions:
                self.generator_to_assumption_estimates[generator_just_used_idx] = generator_just_used.assumptions()

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
                                                                influence, extra_data, v)

            # Grab the assumption estimates, if we're tracking them
            if self.check_assumptions:
                self.generator_to_assumption_estimates[i] = generator.assumptions()

        return generator_to_token_allocs

    def train_aat(self, increase: float, baseline_increase: float, generator_just_used_idx: int) -> None:
        correction_term = increase / baseline_increase
        generator_assumption_estimates = self.generator_to_assumption_estimates[generator_just_used_idx]
        alignment_vector = generator_assumption_estimates.alignment_vector()

        # Store the alignment vector
        file_path = f'../aat/training_data/generator_{generator_just_used_idx}_vectors'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(alignment_vector)

        # Store the correction term
        file_path = f'../aat/training_data/generator_{generator_just_used_idx}_correction_terms'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([correction_term])
