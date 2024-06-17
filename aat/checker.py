from aat.assumptions import Assumptions
from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import List, Set


#   - Determine who to attack (6/12):
#       - Does the agent receive profit from attacking player i?
#       - Does player i receive damage?

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
        self.prev_tokens_kept = None
        self.prev_popularity_k = None
        self.n_tokens = None
        self.prev_attack_tokens = None
        self.prev_attack_type = None
        self.prev_tokens_kept_a = None
        self.prev_popularities_a = None

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from progress checkers ------------------------------------------------------------------
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
        self.prominence_below_avg = 0.5
        self.prominence_above_avg = 0.5
        self.prominence_max_val = 0.5
        self.prominence_rank_val = 0.5
        self.familiarity_below_modularity = 0.5
        self.familiarity_above_modularity = 0.5
        self.prosocial_score = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from keep tokens ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.percent_tokens_kept = 0.5
        self.percent_attackers = 0.5
        self.percent_pop_of_attackers = 0.5
        self.percent_impact_of_attackers = 0.5
        self.tokens_kept_below_stolen = 0.5
        self.tokens_kept_above_stolen = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from attacking other players ------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.my_attack_damaged_other_player = 0.5
        self.my_attack_benefited_me = 0.5
        self.vengence_attack = 0.5
        self.defend_friend_attack = 0.5
        self.pillage_attack = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from give tokens ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.percent_of_players_to_give_to = 0.5
        self.percent_of_friends_who_reciprocate = 0.5

    # Todo: have these be based on only the times the generator was used
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

            self.communities_changes_from_prev = n_changes_from_prev / total_possible_changes
            assert 0 <= self.communities_changes_from_prev <= 1

            # Determine how many differences there are between the "regular" communities and the communities when
            # using a max for the IHN matrix
            for community in ihn_max_communities:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_differences_with_ihn_max += n_differences

            self.communities_diffs_with_ihn_max = n_differences_with_ihn_max / total_possible_changes
            assert 0 <= self.communities_diffs_with_ihn_max <= 1

            # Determine how many differences there are between the "regular" communities and the communities when
            # using a min for the IHP matrix
            for community in ihp_min_communities:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_differences_with_ihp_min += n_differences

            self.communities_diffs_with_ihp_min = n_differences_with_ihp_min / total_possible_changes
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

    def how_many_members_missing(self, desired_community, communities: List[Set[int]]) -> None:
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

    def prominence(self, desired_community, popularities) -> None:
        s = desired_community.s
        group_sum, mx, num_greater = 0, 0.0, 0
        for i in s:
            group_sum += popularities[i]
            if popularities[i] > mx:
                mx = popularities[i]
            if popularities[i] > popularities[self.player_idx]:
                num_greater += 1

        if (group_sum > 0.0) and (len(s) > 1):
            ave_sum = group_sum / len(s)
            avg_val = popularities[self.player_idx] / ave_sum
            max_val = popularities[self.player_idx] / mx
            rank_val = 1 - (num_greater / (len(s) - 1.0))

        else:
            avg_val, max_val, rank_val = 1.0, 1.0, 1.0

        if avg_val < 1.0:
            self.prominence_below_avg = avg_val
            self.prominence_above_avg = 0.0

        else:
            self.prominence_below_avg = 0.0
            self.prominence_above_avg = 1 / avg_val

        self.prominence_max_val, self.prominence_rank_val = max_val, rank_val

    def modularity_vs_familiarity(self, desired_community) -> None:
        modularity, familiarity = desired_community.modularity, desired_community.familiarity

        if familiarity < modularity:
            self.familiarity_below_modularity = familiarity / modularity if modularity > 0 else 0.0
            self.familiarity_above_modularity = 0.0

        else:
            self.familiarity_below_modularity = 0.0
            self.familiarity_above_modularity = modularity / familiarity if familiarity > 0 else 0.0

    def prosocial(self, desired_community) -> None:
        self.prosocial_score = desired_community.prosocial

    # ------------------------------------------------------------------------------------------------------------------
    # Keep tokens checkers ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def keep_tokens(self, tokens_kept: int, received: np.array, popularities: np.array, v: np.array) -> None:
        if self.n_tokens is None:
            self.n_tokens = self.n_players * 2

        received_adjusted = received * self.n_tokens

        if self.prev_tokens_kept is not None:
            tokens_stolen, n_attackers, impact_of_attackers = 0, 0, 0
            pop_sum, cumulative_pop_of_attackers = sum(popularities), 0

            for i, tokens_received in enumerate(list(received_adjusted)):
                if tokens_received < 0 and i != self.player_idx:
                    tokens_stolen += abs(tokens_received)
                    n_attackers += 1
                    impact_of_attackers += abs(v[i][self.player_idx])
                    cumulative_pop_of_attackers += popularities[i]

            self.percent_tokens_kept = self.prev_tokens_kept / self.n_tokens
            self.percent_attackers = n_attackers / self.n_players
            self.percent_pop_of_attackers = cumulative_pop_of_attackers / pop_sum
            self.percent_impact_of_attackers = min(impact_of_attackers / self.prev_popularity_k, 1.0) \
                if self.prev_popularity_k > 0 else 0.0

            if self.prev_tokens_kept < tokens_stolen:
                self.tokens_kept_below_stolen = self.prev_tokens_kept / tokens_stolen if tokens_stolen > 0 else 0.0
                self.tokens_kept_above_stolen = 0.0

            else:
                self.tokens_kept_below_stolen = 0.0
                self.tokens_kept_above_stolen = tokens_stolen / self.prev_tokens_kept if self.prev_tokens_kept > 0 \
                    else 0.0

        self.prev_tokens_kept = tokens_kept
        self.prev_popularity_k = popularities[self.player_idx]

    # ------------------------------------------------------------------------------------------------------------------
    # Attack other players checkers ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def attack_was_successful(self, attack_tokens: np.array, v: np.array, tokens_kept: int,
                              popularities: np.array) -> None:

        if self.prev_attack_tokens is not None:
            player_that_was_attacked, tokens_stolen = None, None

            for i, tokens in enumerate(list(self.prev_attack_tokens)):
                if tokens > 0:
                    player_that_was_attacked, tokens_stolen = i, tokens
                    break

            if player_that_was_attacked is not None:
                my_impact_on_that_player = abs(v[self.player_idx][player_that_was_attacked])
                my_popularity, their_popularity = \
                    self.prev_popularities_a[self.player_idx], self.prev_popularities_a[player_that_was_attacked]
                percent_stolen = tokens_stolen / (tokens_stolen + self.prev_tokens_kept_a)
                my_benefit = v[self.player_idx][self.player_idx] * percent_stolen

                self.my_attack_damaged_other_player = min(my_impact_on_that_player / their_popularity, 1.0) \
                    if their_popularity > 0 else 0.0
                self.my_attack_benefited_me = min(my_benefit / my_popularity, 1.0) if my_popularity > 0 else 0.0

            else:
                self.my_attack_damaged_other_player = 0.0
                self.my_attack_benefited_me = 0.0

        self.prev_attack_tokens = attack_tokens
        self.prev_tokens_kept_a = tokens_kept
        self.prev_popularities_a = popularities

    def attack_type(self, attack_type: str) -> None:
        if self.prev_attack_type is not None:
            if self.prev_attack_type == 'v':
                self.vengence_attack = 1.0
                self.defend_friend_attack = 0.0
                self.pillage_attack = 0.0

            elif self.prev_attack_type == 'df':
                self.vengence_attack = 0.0
                self.defend_friend_attack = 1.0
                self.pillage_attack = 0.0

            elif self.prev_attack_type == 'p':
                self.vengence_attack = 0.0
                self.defend_friend_attack = 0.0
                self.pillage_attack = 1.0

            else:
                self.vengence_attack = 0.0
                self.defend_friend_attack = 0.0
                self.pillage_attack = 0.0

        self.prev_attack_type = attack_type

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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

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
            self.prominence_below_avg,
            self.prominence_above_avg,
            self.prominence_max_val,
            self.prominence_rank_val,
            self.familiarity_below_modularity,
            self.familiarity_above_modularity,
            self.prosocial_score,
            self.percent_tokens_kept,
            self.percent_attackers,
            self.percent_pop_of_attackers,
            self.percent_impact_of_attackers,
            self.tokens_kept_below_stolen,
            self.tokens_kept_above_stolen,
            self.my_attack_damaged_other_player,
            self.my_attack_benefited_me,
            self.vengence_attack,
            self.defend_friend_attack,
            self.pillage_attack,
            self.percent_of_players_to_give_to,
            self.percent_of_friends_who_reciprocate
        )

        return estimated_assumptions
