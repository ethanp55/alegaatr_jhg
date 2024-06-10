from aat.assumptions import Assumptions
from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import List, Set


#   - Determine desired community (these pertain to the specific community chosen by the CAB agent) (6/11):
#       - 3 - Have there been recent significant changes in prominence?
#       - 4 - Have there been recent significant changes in familiarity scores?
#       - 5 - Have there been recent significant changes in prosocial behavior?
#       - 6 - What are the different weights (maybe)?


# Class that contains all the alignment checkers
class AssumptionChecker:
    def __init__(self) -> None:
        # Helper variables used to estimate assumptions
        self.prev_popularity, self.prev_rank = None, None
        self.medium_term_popularities, self.long_term_popularities = deque(maxlen=2), deque(maxlen=4)
        self.medium_term_ranks, self.long_term_ranks = deque(maxlen=2), deque(maxlen=4)
        self.prev_modularities = deque(maxlen=5)
        self.prev_communities = None
        self.n_players, self.player_idx = None, None
        self.prev_collective_strength = None

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
        self.communities_changes_from_prev = 0.5
        self.communities_diffs_with_ihn_max = 0.5
        self.communities_diffs_with_ihp_min = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from determine desired community --------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.below_prev_collective_strength = 0.5
        self.above_prev_collective_strength = 0.5
        self.below_target_strength = 0.5
        self.above_target_strength = 0.5
        self.percent_of_players_needed_for_desired_community = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from give tokens ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.percent_of_players_to_give_to = 0.5
        self.percent_of_friends_who_reciprocate = 0.5

    # ------------------------------------------------------------------------------------------------------------------
    # Progress checkers ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def progress_checkers(self, player_idx: int, round_num: int, popularities: np.array) -> None:
        if self.n_players is None:
            self.n_players = len(popularities)
            self.player_idx = player_idx

        curr_popularity = popularities[player_idx]
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
        if self.n_players <= 10:
            self.below_10_players = self.n_players / 10
            self.above_10_players = 0.0

        else:
            self.below_10_players = 0.0
            self.above_10_players = 10 / self.n_players

    # ------------------------------------------------------------------------------------------------------------------
    # Detect communities checkers --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def graph_connectedness(self, influence_matrix: np.array) -> None:
        # Calculate the density for positive edges and the density for negative edges; filter any small values
        positive_influence = np.where(influence_matrix < 0, 0, influence_matrix)
        negative_influence = np.abs(np.where(influence_matrix > 0, 0, influence_matrix))
        max_positive_influence = np.max(positive_influence)
        max_negative_influence = np.max(negative_influence)
        n_positive_edges = np.sum(
            (positive_influence / max_positive_influence) >= 0.05) if max_positive_influence > 0 else 0
        n_negative_edges = np.sum(
            (negative_influence / max_negative_influence) >= 0.05) if max_negative_influence > 0 else 0
        max_edges = self.n_players ** 2

        self.positive_density = n_positive_edges / max_edges
        self.negative_density = n_negative_edges / max_edges

    def graph_edge_percentages(self, influence_matrix: np.array) -> None:
        # Calculate the percentage of positive edges, neutral (0) edges, and negative edges
        max_edges = self.n_players ** 2
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

    def changes_in_communities(self, communities: List[Set[int]], ihn_max_communities: List[Set[int]],
                               ihp_min_communities: List[Set[int]]) -> None:
        if self.prev_communities is not None:
            n_changes_from_prev, n_differences_with_ihn_max, n_differences_with_ihp_min = 0, 0, 0
            total_possible_changes = (self.n_players - 1) * self.n_players
            player_to_community = {}

            # Determine how many community changes, relative to the total number of players, there are compared to the
            # previous round
            for community in communities:
                for player in community:
                    player_prev_community = self.prev_communities[player]
                    n_differences = len(player_prev_community.symmetric_difference(community))
                    n_changes_from_prev += n_differences
                    player_to_community[player] = community

            self.communities_changes_from_prev = 1 - (n_changes_from_prev / total_possible_changes)
            assert 0 <= self.communities_changes_from_prev <= 1

            # Determine how many differences there are between the "regular" communities and the communities when
            # using a max for the IHN matrix
            for community in ihn_max_communities:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_differences_with_ihn_max += n_differences

            self.communities_diffs_with_ihn_max = 1 - (n_differences_with_ihn_max / total_possible_changes)
            assert 0 <= self.communities_diffs_with_ihn_max <= 1

            # Determine how many differences there are between the "regular" communities and the communities when
            # using a min for the IHP matrix
            for community in ihp_min_communities:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_differences_with_ihp_min += n_differences

            self.communities_diffs_with_ihp_min = 1 - (n_differences_with_ihp_min / total_possible_changes)
            assert 0 <= self.communities_diffs_with_ihp_min <= 1

        self.prev_communities = {}
        for community in communities:
            for player in community:
                self.prev_communities[player] = community

    # ------------------------------------------------------------------------------------------------------------------
    # Determine desired community checkers -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def changes_in_collective_strength(self, desired_community) -> None:
        collective_strength = desired_community.collective_strength

        if self.prev_collective_strength is not None:
            if collective_strength < self.prev_collective_strength:
                self.below_prev_collective_strength = collective_strength / self.prev_collective_strength
                self.above_prev_collective_strength = 0.0

            else:
                self.below_prev_collective_strength = 0.0
                self.above_prev_collective_strength = self.prev_collective_strength / collective_strength

        self.prev_collective_strength = collective_strength

    def how_close_to_target_strength(self, desired_community, target) -> None:
        collective_strength = desired_community.collective_strength

        if collective_strength < target:
            self.below_target_strength = collective_strength / target
            self.above_target_strength = 0.0

        else:
            self.below_target_strength = 0.0
            self.above_target_strength = target / collective_strength

    def how_many_members_missing(self, desired_community, communities: List[Set[int]]):
        desired_group = desired_community.s
        player_group = None

        for community in communities:
            if self.player_idx in community:
                player_group = community
                break

        assert player_group is not None
        n_differences = len(player_group.symmetric_difference(desired_group))
        assert self.n_players >= n_differences

        self.percent_of_players_needed_for_desired_community = n_differences / self.n_players

    # ------------------------------------------------------------------------------------------------------------------
    # Give tokens checkers ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def percentage_of_players_to_give_to(self, token_allocations: np.array) -> None:
        n_friends = 0

        for i in range(self.n_players):
            n_friends += 1 if (i != self.player_idx and token_allocations[i] > 0) else 0

        self.percent_of_players_to_give_to = n_friends / self.n_players

    def friends_are_reciprocating(self, influence_matrix: np.array, token_allocations: np.array) -> None:
        friend_indices = list(np.where(token_allocations > 0)[0])
        n_friends, n_friends_who_reciprocate = len(friend_indices), 0

        for friend_idx in friend_indices:
            if friend_idx == self.player_idx:
                continue
            my_influence_on_friend = influence_matrix[self.player_idx][friend_idx]
            friends_influence_on_me = influence_matrix[friend_idx][self.player_idx]
            n_friends_who_reciprocate += 1 if friends_influence_on_me >= my_influence_on_friend else 0

        self.percent_of_friends_who_reciprocate = n_friends_who_reciprocate / n_friends if n_friends > 0 else 0.0

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
            self.modularity_below_ema,
            self.communities_changes_from_prev,
            self.communities_diffs_with_ihn_max,
            self.communities_diffs_with_ihp_min,
            self.below_prev_collective_strength,
            self.above_prev_collective_strength,
            self.below_target_strength,
            self.above_target_strength,
            self.percent_of_players_needed_for_desired_community,

            self.percent_of_players_to_give_to,
            self.percent_of_friends_who_reciprocate
        )

        return estimated_assumptions
