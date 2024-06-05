from aat.assumptions import Assumptions
from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


# Class that contains all the alignment checkers
class AssumptionChecker:
    def __init__(self) -> None:
        # Helper variables used to estimate assumptions
        self.prev_popularity, self.prev_rank = None, None
        self.medium_term_popularities, self.long_term_popularities = deque(maxlen=2), deque(maxlen=4)
        self.medium_term_ranks, self.long_term_ranks = deque(maxlen=2), deque(maxlen=4)
        self.prev_modularities = deque(maxlen=5)

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from progress checkers  -----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.improved_from_prev_round = 0.5
        self.improved_medium_term = 0.5
        self.improved_long_term = 0.5
        self.rank = 0.5
        self.rank_improved_from_prev_round = 0.5
        self.rank_improved_medium_term = 0.5
        self.rank_improved_long_term = 0.5
        self.percentile = 0.5
        self.below_30_rounds = 0.5
        self.above_30_rounds = 0.5
        self.below_10_players = 0.5
        self.above_10_players = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from detect communities -----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.positive_density = 0.5
        self.negative_density = 0.5
        self.percentage_of_positive_edges = 0.5
        self.percentage_of_neutral_edges = 0.5
        self.percentage_of_negative_edges = 0.5
        self.modularity_above_ema = 0.5
        self.modularity_below_ema = 0.5

    # ------------------------------------------------------------------------------------------------------------------
    # Progress checkers ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def progress_checkers(self, player_idx: int, round_num: int, popularities: np.array) -> None:
        curr_popularity, n_players = popularities[player_idx], len(popularities)
        round_num_adjusted = round_num + 1  # Round numbers are 0-indexed by default

        # Assumption about whether our popularity is better than previous round (short-term)
        if self.prev_popularity is not None:
            self.improved_from_prev_round = float(curr_popularity > self.prev_popularity)
        self.prev_popularity = curr_popularity

        # Assumption about whether our popularity is better than 2 rounds ago (medium-term)
        if len(self.medium_term_popularities) == 2:
            self.improved_medium_term = float(curr_popularity > self.medium_term_popularities[0])
        self.medium_term_popularities.append(curr_popularity)

        # Assumption about whether our popularity is better than 4 rounds ago (long-term)
        if len(self.long_term_popularities) == 4:
            self.improved_long_term = float(curr_popularity > self.long_term_popularities[0])
        self.long_term_popularities.append(curr_popularity)

        relative_popularity = curr_popularity / sum(popularities)

        # Assumption about our relative popularity
        self.rank = relative_popularity

        # Assumption about whether our rank is better than previous round (short-term)
        if self.prev_rank is not None:
            self.rank_improved_from_prev_round = float(relative_popularity > self.prev_rank)
        self.prev_rank = relative_popularity

        # Assumption about whether our rank is better than 2 rounds ago (medium-term)
        if len(self.medium_term_ranks) == 2:
            self.rank_improved_medium_term = float(relative_popularity > self.medium_term_ranks[0])
        self.medium_term_ranks.append(relative_popularity)

        # Assumption about whether our rank is better than 4 rounds ago (medium-term)
        if len(self.long_term_ranks) == 4:
            self.rank_improved_long_term = float(relative_popularity > self.long_term_ranks[0])
        self.long_term_ranks.append(relative_popularity)

        percentile = percentileofscore(popularities, curr_popularity, kind='rank')

        # Assumption about what percentile we're in
        self.percentile = float(percentile / 100)

        # Assumptions about whether we are below/above 30 rounds
        if round_num_adjusted <= 30:
            self.below_30_rounds = round_num_adjusted / 30
            self.above_30_rounds = 0.0

        else:
            self.below_30_rounds = 0.0
            self.above_30_rounds = 30 / round_num_adjusted

        # Assumptions about whether we are below/above 10 players
        if n_players <= 10:
            self.below_10_players = n_players / 10
            self.above_10_players = 0.0

        else:
            self.below_10_players = 0.0
            self.above_10_players = 10 / n_players

    # ------------------------------------------------------------------------------------------------------------------
    # Detect communities checkers --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def graph_connectedness(self, influence_matrix: np.array) -> None:
        # Calculate the density for positive edges and the density for negative edges; filter any small values
        positive_influence = np.where(influence_matrix < 0, 0, influence_matrix)
        negative_influence = np.abs(np.where(influence_matrix > 0, 0, influence_matrix))
        max_positive_influence = np.max(positive_influence)
        max_negative_influence = np.max(negative_influence)
        n_players = len(influence_matrix)
        n_positive_edges = np.sum(
            (positive_influence / max_positive_influence) >= 0.05) if max_positive_influence > 0 else 0
        n_negative_edges = np.sum(
            (negative_influence / max_negative_influence) >= 0.05) if max_negative_influence > 0 else 0
        max_edges = n_players ** 2

        self.positive_density = n_positive_edges / max_edges
        self.negative_density = n_negative_edges / max_edges

    def graph_edge_percentages(self, influence_matrix: np.array) -> None:
        # Calculate the percentage of positive edges, neutral (0) edges, and negative edges
        n_players = len(influence_matrix)
        max_edges = n_players ** 2
        n_positive_edges = np.sum(influence_matrix > 0.0)
        n_neutral_edges = np.sum(influence_matrix == 0.0)

        self.percentage_of_positive_edges = n_positive_edges / max_edges
        self.percentage_of_neutral_edges = n_neutral_edges / max_edges
        self.percentage_of_negative_edges = 1 - (self.percentage_of_positive_edges + self.percentage_of_neutral_edges)

        assert sum([self.percentage_of_positive_edges, self.percentage_of_neutral_edges,
                    self.percentage_of_negative_edges]) == 1.0

    def changes_in_modularity(self, modularity: float) -> None:
        curr_ema = pd.Series.ewm(pd.Series(self.prev_modularities), span=5).mean()

        if len(curr_ema) > 0:
            curr_ema_val = curr_ema.iloc[-1,]

            if modularity > curr_ema_val:
                self.modularity_above_ema = curr_ema_val / modularity
                self.modularity_below_ema = 0.0

            else:
                self.modularity_above_ema = 0.0
                self.modularity_below_ema = modularity / curr_ema_val

        self.prev_modularities.append(modularity)

    def assumptions(self) -> Assumptions:
        estimated_assumptions = Assumptions(
            self.improved_from_prev_round,
            self.improved_medium_term,
            self.improved_long_term,
            self.rank,
            self.rank_improved_from_prev_round,
            self.rank_improved_medium_term,
            self.rank_improved_long_term,
            self.percentile,
            self.below_30_rounds,
            self.above_30_rounds,
            self.below_10_players,
            self.above_10_players,
            self.positive_density,
            self.negative_density,
            self.percentage_of_positive_edges,
            self.percentage_of_neutral_edges,
            self.percentage_of_negative_edges,
            self.modularity_above_ema,
            self.modularity_below_ema
        )

        return estimated_assumptions
