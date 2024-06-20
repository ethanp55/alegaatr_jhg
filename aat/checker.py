from aat.assumptions import Assumptions
from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import Dict, List, Optional, Set


class AssumptionChecker:
    def __init__(self) -> None:
        # Helper variables used to estimate assumptions
        self.player_idx, self.n_players, self.n_tokens, self.game_params = None, None, None, None
        self.round_previously_used = None
        self.prev_modularities = deque(maxlen=5)
        self.prev_communities = None
        self.communities_from_last_use = None
        self.prev_desired_comm = None
        self.desired_comm_from_last_use = None
        self.prev_collective_strength = None
        self.prev_popularity = None
        self.prev_tokens_kept = None
        self.prev_popularities_a = None
        self.round_previously_used_a = None
        self.pop_before_last_attack = None
        self.prev_attack_tokens_used = None
        self.prev_attack_gain_pred, self.attack_damage_pred, self.prev_player_to_attack = None, None, None
        self.prev_popularities_a_pred = None
        self.players_that_have_attacked = {}
        self.prev_tokens_that_were_given = None
        self.round_previously_used_g = None
        self.our_influence_last_time, self.their_influence_within_2 = None, None

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from progress checkers ------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # 13 total
        self.pop_improved_1_round_after = 0.5
        self.pop_improved_2_rounds_after = 0.5
        self.rel_pop_improved_1_round_after = 0.5
        self.rel_pop_improved_2_rounds_after = 0.5
        self.rank_1_round_after = 0.5
        self.rank_2_rounds_after = 0.5
        self.curr_rel_pop = 0.5
        self.curr_rank = 0.5
        self.below_30_rounds = 0.5
        self.above_30_rounds = 0.5
        self.below_10_players = 0.5
        self.above_10_players = 0.5
        self.was_just_used = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from detect communities -----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # 9 total
        self.positive_density = 0.5
        self.negative_density = 0.5
        self.modularity_above_ema = 0.5
        self.modularity_below_ema = 0.5
        self.communities_changes_from_prev = 0.5
        self.communities_diffs_with_ihn_max = 0.5
        self.communities_diffs_with_ihp_min = 0.5
        self.communities_diffs_with_just_used = 0.5
        self.communities_diffs_from_last_use = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from determine desired community --------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # 11 total
        self.collective_strength_increased = 0.5
        self.community_has_significant_strength = 0.5
        self.near_target_strength = 0.5
        self.percent_of_players_needed_for_desired_community = 0.5
        self.prominence_avg_val = 0.5
        self.prominence_max_val = 0.5
        self.prominence_rank_val = 0.5
        self.familiarity_better_than_modularity = 0.5
        self.prosocial_score = 0.5
        self.desired_comm_diffs_with_just_used = 0.5
        self.desired_comm_diffs_from_last_use = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from keep tokens ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # 8 total
        self.does_not_keep_too_much = 0.5
        self.n_attackers_is_low = 0.5
        self.attackers_are_weak = 0.5
        self.defense_was_effective = 0.5
        self.defense_was_effective_last_time = 0.5
        self.defense_would_have_been_effective = 0.5
        self.none_in_desired_community = 0.5
        self.none_in_existing_community = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from attacking other players ------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # 10 total
        self.my_attack_damaged_other_player = 0.5
        self.my_attack_benefited_me = 0.5
        self.pop_did_not_decrease_after_attack = 0.5
        self.attack_would_have_damaged_other = 0.5
        self.attack_would_have_benefited_us = 0.5
        self.does_not_attack_too_much = 0.5
        self.attacked_player_not_in_community = 0.5
        self.attacked_player_not_in_desired_group = 0.5
        self.attack_damaged_other_player = 0.5
        self.attack_benefited_me = 0.5

        # --------------------------------------------------------------------------------------------------------------
        # Assumption estimates from give tokens ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # 7 total
        self.does_not_give_too_much = 0.5
        self.gives_to_all_players_in_desired_group = 0.5
        self.all_friends_reciprocate = 0.5
        self.no_friends_have_attacked = 0.5
        self.no_friends_have_attacked_us = 0.5
        self.given_to_all_in_desired_group = 0.5
        self.all_friends_reciprocated_within_2_last_time = 0.5

    # Function for initializing static variables used in the checker calculations
    def init_vars(self, player_idx: int, n_players: int, n_tokens: int, game_params: Dict[str, float]) -> None:
        self.player_idx = player_idx
        self.n_players = n_players
        self.n_tokens = n_tokens
        self.game_params = game_params

    # ------------------------------------------------------------------------------------------------------------------
    # Progress checkers ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def popularity_increased(self, round_num: int, was_just_used: bool, pop_history: List[np.array]) -> None:
        if was_just_used:
            self.round_previously_used = round_num

        self.was_just_used = float(was_just_used)

        # Check for changes in popularity (absolute and relative) and percentile/rank 1 and 2 rounds after last using
        # the CAB agent, if possible; default to 0.5 (to represent uncertainty) if we cannot calculate
        if self.round_previously_used is not None:
            round_2 = self.round_previously_used
            round_1, round_3 = round_2 - 1, round_2 + 1

            if round_1 >= 0:
                round_1_pops, round_2_pops = pop_history[round_1], pop_history[round_2]
                round_1_pop, round_2_pop = round_1_pops[self.player_idx], round_2_pops[self.player_idx]
                round_1_sum, round_2_sum = sum(round_1_pops), sum(round_2_pops)
                round_1_relative, round_2_relative = round_1_pop / round_1_sum, round_2_pop / round_2_sum
                round_2_percentile = percentileofscore(round_2_pops, round_2_pop, kind='rank')

                self.pop_improved_1_round_after = float(round_2_pop > round_1_pop)
                self.rel_pop_improved_1_round_after = float(round_2_relative > round_1_relative)
                self.rank_1_round_after = round_2_percentile

            else:
                self.pop_improved_1_round_after = 0.5
                self.rel_pop_improved_1_round_after = 0.5
                self.rank_1_round_after = 0.5

            if round_1 >= 0 and round_3 < len(pop_history):
                round_1_pops, round_3_pops = pop_history[round_1], pop_history[round_3]
                round_1_pop, round_3_pop = round_1_pops[self.player_idx], round_3_pops[self.player_idx]
                round_1_sum, round_3_sum = sum(round_1_pops), sum(round_3_pops)
                round_1_relative, round_3_relative = round_1_pop / round_1_sum, round_3_pop / round_3_sum
                round_3_percentile = percentileofscore(round_3_pops, round_3_pop, kind='rank')

                self.pop_improved_2_rounds_after = float(round_3_pop > round_1_pop)
                self.rel_pop_improved_2_rounds_after = float(round_3_relative > round_1_relative)
                self.rank_2_rounds_after = round_3_percentile

            else:
                self.pop_improved_2_rounds_after = 0.5
                self.rel_pop_improved_2_rounds_after = 0.5
                self.rank_2_rounds_after = 0.5

    def current_values(self, round_num: int, popularities: np.array) -> None:
        curr_pop = popularities[self.player_idx]
        round_num_adjusted = round_num + 1  # Round numbers are 0-indexed by default

        # Current relative popularity and percentile/rank
        self.curr_rel_pop = curr_pop / sum(popularities)
        self.curr_rank = percentileofscore(popularities, curr_pop, kind='rank')

        # Checker how many rounds below/above 30 we are
        if round_num_adjusted <= 30:
            self.below_30_rounds = round_num_adjusted / 30
            self.above_30_rounds = 0.0

        else:
            self.below_30_rounds = 0.0
            self.above_30_rounds = 30 / round_num_adjusted

    def society_size(self) -> None:
        # Check how many players below/above 10 there are
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

    def changes_in_modularity(self, modularity: float) -> None:
        # Determine how far above or below the current modularity is from the EMA 5 value (i.e., check for any sudden
        # increases/decreases compared to the recent average)
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
                               ihp_min_communities: List[Set[int]], was_just_used: bool,
                               communities_just_used: Optional[List[Set[int]]]) -> None:
        if was_just_used and self.prev_communities is not None:
            self.communities_from_last_use = self.prev_communities

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

            # Determine how many differences there are between the "regular" communities and the communities when
            # using a max for the IHN matrix
            for community in ihn_max_communities:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_differences_with_ihn_max += n_differences

            self.communities_diffs_with_ihn_max = 1 - (n_differences_with_ihn_max / total_possible_changes)

            # Determine how many differences there are between the "regular" communities and the communities when
            # using a min for the IHP matrix
            for community in ihp_min_communities:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_differences_with_ihp_min += n_differences

            self.communities_diffs_with_ihp_min = 1 - (n_differences_with_ihp_min / total_possible_changes)

        self.prev_communities = {}
        for community in communities:
            for player in community:
                self.prev_communities[player] = community

        if communities_just_used is not None:
            # Determine how many community changes, relative to the total number of players, there are compared to the
            # detected communities of the agent that was just used
            n_changes_from_curr, total_possible_changes = 0, (self.n_players - 1) * self.n_players
            player_to_community = {}

            for community in communities:
                for player in community:
                    player_to_community[player] = community

            for community in communities_just_used:
                for player in community:
                    regular_community = player_to_community[player]
                    n_differences = len(regular_community.symmetric_difference(community))
                    n_changes_from_curr += n_differences

            self.communities_diffs_with_just_used = 1 - (n_changes_from_curr / total_possible_changes)

        if self.communities_from_last_use is not None:
            # Determine how many community changes, relative to the total number of players, there are compared to the
            # last time the agent was used
            n_changes_from_prev, total_possible_changes = 0, (self.n_players - 1) * self.n_players

            for community in communities:
                for player in community:
                    player_prev_community = self.communities_from_last_use[player]
                    n_differences = len(player_prev_community.symmetric_difference(community))
                    n_changes_from_prev += n_differences

            self.communities_diffs_from_last_use = 1 - (n_changes_from_prev / total_possible_changes)

    # ------------------------------------------------------------------------------------------------------------------
    # Determine desired community checkers -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def collective_strength(self, desired_community) -> None:
        strength = desired_community.collective_strength

        # Check if the collective strength increased from the previous round
        if self.prev_collective_strength is not None:
            self.collective_strength_increased = float(strength > self.prev_collective_strength)

        # Compare the strength of the desired community with the total popularity of society (i.e., how much of a
        # presence the community has)
        self.community_has_significant_strength = strength

        self.prev_collective_strength = strength

    def close_to_target_strength(self, desired_community, target) -> None:
        # Calculate how far away the collective strength is from the target strength
        collective_strength = desired_community.collective_strength
        self.near_target_strength = min(collective_strength / target, 1.0)

    def how_many_members_missing(self, desired_community, communities: List[Set[int]]) -> None:
        # Calculate how many members still need to be added/removed to form the desired community
        desired_group = set(desired_community.s)
        player_group = None

        for community in communities:
            if self.player_idx in community:
                player_group = community
                break

        n_differences, total_possible_diffs = len(player_group.symmetric_difference(desired_group)), self.n_players - 1

        self.percent_of_players_needed_for_desired_community = n_differences / total_possible_diffs

    def prominence(self, desired_community, popularities) -> None:
        # Calculate the prominence scores
        s = set(desired_community.s)
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

        self.prominence_max_val, self.prominence_rank_val, self.prominence_avg_val = \
            max_val, rank_val, min(avg_val, 1.0)

    def modularity_vs_familiarity(self, desired_community) -> None:
        # Calculate whether the familiarity is better than modularity
        modularity, familiarity = desired_community.modularity, desired_community.familiarity
        self.familiarity_better_than_modularity = float(familiarity > modularity)

    def prosocial(self, desired_community) -> None:
        # Calculate how good the prosocial score is
        self.prosocial_score = desired_community.prosocial

    def desired_community_differences(self, desired_community, was_just_used: bool,
                                      desired_community_just_used: Optional) -> None:
        s = set(desired_community.s)

        if was_just_used and self.prev_desired_comm is not None:
            self.desired_comm_from_last_use = self.prev_desired_comm

        # Calculate the number of differences between the desired community and the desired community that was just used
        if desired_community_just_used is not None:
            total_possible_diffs = self.n_players - 1
            s_just_used = set(desired_community_just_used.s)
            n_differences = len(s.symmetric_difference(s_just_used))

            self.desired_comm_diffs_with_just_used = 1 - (n_differences / total_possible_diffs)

        # Calculate the number of differences between the desired community and the desired community from the last time
        # the agent was used
        if self.desired_comm_from_last_use is not None:
            total_possible_diffs = self.n_players - 1
            s_from_last_use = self.desired_comm_from_last_use
            n_differences = len(s.symmetric_difference(s_from_last_use))

            self.desired_comm_diffs_from_last_use = 1 - (n_differences / total_possible_diffs)

        self.prev_desired_comm = s

    # ------------------------------------------------------------------------------------------------------------------
    # Keep tokens checkers ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def n_tokens_kept(self, tokens_kept: int) -> None:
        self.does_not_keep_too_much = 1 - (tokens_kept / self.n_tokens)

    def attackers(self, received: np.array, popularities: np.array, desired_community,
                  communities: List[Set[int]]) -> None:
        attacker_indices = []
        pop_sum, cumulative_pop_of_attackers = sum(popularities), 0

        for i, tokens_received in enumerate(list(received)):
            if tokens_received < 0 and i != self.player_idx:
                attacker_indices.append(i)
                cumulative_pop_of_attackers += popularities[i]

        n_attackers = len(attacker_indices)

        # Calculate how many attackers and how strong the attackers are
        self.n_attackers_is_low = 1 - (n_attackers / self.n_players)
        self.attackers_are_weak = 1 - (cumulative_pop_of_attackers / pop_sum)

        # Calculate how many attackers are in the CAB's desired and detected/current communities
        desired_group = set(desired_community.s)
        player_group = None
        n_in_desired, n_in_current = 0, 0

        for community in communities:
            if self.player_idx in community:
                player_group = community
                break

        for i in attacker_indices:
            n_in_desired += 1 if i in desired_group else 0
            n_in_current += 1 if i in player_group else 0

        self.none_in_desired_community = 1 - (n_in_desired / len(desired_group))
        self.none_in_existing_community = 1 - (n_in_current / len(player_group))

    def defense_strength(self, received: np.array, popularities: np.array, tokens_kept: int, was_used: bool) -> None:
        impact_with_no_tokens, true_impact, what_if_impact, c_take = 0, 0, 0, self.game_params['steal']
        pop = popularities[self.player_idx]

        # Calculate how effective the defense was by comparing what would have happened if no tokens were kept
        for i, tokens_received in enumerate(list(received)):
            if tokens_received < 0 and i != self.player_idx:
                steal_impact = c_take * -received[i]
                impact_with_no_tokens += steal_impact

        for i, tokens_received in enumerate(list(received)):
            if tokens_received < 0 and i != self.player_idx:
                numerator = pop * received[self.player_idx]
                denominator = sum([popularities[j] * -received[j] for j in range(len(received)) if
                                   received[j] < 0 and j != self.player_idx])
                steal_impact = c_take * max(0, 1 - (numerator / denominator)) * -received[i]
                true_impact += steal_impact

        self.defense_was_effective = (1 - (true_impact / impact_with_no_tokens)) if impact_with_no_tokens != 0 else 0.5

        # Calculate how effective the CAB's defense was the last time it was used
        if was_used and impact_with_no_tokens != 0:
            self.defense_was_effective_last_time = self.defense_was_effective

        # Calculate how effective the CAB's defense would have been if it had been used
        if self.prev_tokens_kept is not None:
            proportion_kept = self.prev_tokens_kept / self.n_tokens

            for i, tokens_received in enumerate(list(received)):
                if tokens_received < 0 and i != self.player_idx:
                    numerator = pop * proportion_kept
                    denominator = sum([popularities[j] * -received[j] for j in range(len(received)) if
                                       received[j] < 0 and j != self.player_idx])
                    steal_impact = c_take * max(0, 1 - (numerator / denominator)) * -received[i]
                    what_if_impact += steal_impact

            self.defense_would_have_been_effective = (
                    1 - (what_if_impact / impact_with_no_tokens)) if impact_with_no_tokens != 0 else 0.5

        self.prev_tokens_kept = tokens_kept

    # ------------------------------------------------------------------------------------------------------------------
    # Attack other players checkers ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def attack_results(self, attack_tokens_used: np.array, v: np.array, popularities: np.array, was_used: bool,
                       round_num: int, received: np.array, desired_community, communities: List[Set[int]]) -> None:
        if self.prev_attack_tokens_used is not None:
            tokens_kept = received[self.player_idx] * self.n_tokens
            player_that_was_attacked, tokens_stolen = None, 0
            for i, tokens in enumerate(list(self.prev_attack_tokens_used)):
                if tokens > 0:
                    player_that_was_attacked, tokens_stolen = i, tokens
                    break

            # Calculate how effective the CAB's attack was the last time it was used (if there was an attack)
            if was_used and tokens_stolen > 0:
                self.round_previously_used_a = round_num
                self.pop_before_last_attack = self.prev_popularities_a[self.player_idx]

                my_impact_on_that_player = abs(v[self.player_idx][player_that_was_attacked])
                my_popularity, their_popularity = \
                    self.prev_popularities_a[self.player_idx], self.prev_popularities_a[player_that_was_attacked]
                percent_stolen = tokens_stolen / (tokens_stolen + tokens_kept)
                my_benefit = v[self.player_idx][self.player_idx] * percent_stolen

                self.my_attack_damaged_other_player = min(my_impact_on_that_player / their_popularity, 1.0) \
                    if their_popularity > 0 else 0.0
                self.my_attack_benefited_me = min(my_benefit / my_popularity, 1.0) if my_popularity > 0 else 0.0

            # Calculate whether our popularity decreased 2 rounds after the CAB was last used and attacked another
            # player
            if self.round_previously_used_a is not None and round_num == self.round_previously_used_a + 1:
                my_popularity = popularities[self.player_idx]
                self.pop_did_not_decrease_after_attack = min(1.0, my_popularity / self.pop_before_last_attack) \
                    if self.pop_before_last_attack > 0 else 0.5

            # Calculate how effective the previous agent's attack was
            if tokens_stolen > 0:
                my_impact_on_that_player = abs(v[self.player_idx][player_that_was_attacked])
                my_popularity, their_popularity = \
                    self.prev_popularities_a[self.player_idx], self.prev_popularities_a[player_that_was_attacked]
                percent_stolen = tokens_stolen / (tokens_stolen + tokens_kept)
                my_benefit = v[self.player_idx][self.player_idx] * percent_stolen

                self.attack_damaged_other_player = min(my_impact_on_that_player / their_popularity, 1.0) \
                    if their_popularity > 0 else 0.0
                self.attack_benefited_me = min(my_benefit / my_popularity, 1.0) if my_popularity > 0 else 0.0

            # Calculate whether the attacked player, if one exists, was in the CAB's detected community
            player_group = None

            for community in communities:
                if self.player_idx in community:
                    player_group = community
                    break

            self.attacked_player_not_in_community = float(
                player_that_was_attacked not in player_group) if player_that_was_attacked is not None else 1.0

            # Calculate whether the attacked player, if one exists, was in the CAB's desired community
            desired_group = set(desired_community.s)
            self.attacked_player_not_in_desired_group = float(
                player_that_was_attacked not in desired_group) if player_that_was_attacked is not None else 1.0

        self.prev_popularities_a = popularities
        self.prev_attack_tokens_used = attack_tokens_used

    def attack_predictions(self, n_tokens: int, gain_per_token: float, damage: float, player_to_attack: int,
                           popularities: np.array) -> None:
        gain = n_tokens * gain_per_token

        if self.prev_attack_gain_pred is not None:
            my_pop = self.prev_popularities_a_pred[self.player_idx]
            their_pop = self.prev_popularities_a_pred[self.prev_player_to_attack]

            self.attack_would_have_benefited_us = max(0.0, min(1.0, self.prev_attack_gain_pred / my_pop)) \
                if my_pop > 0 else 0.0
            self.attack_would_have_damaged_other = max(0.0, min(1.0, damage / their_pop)) if their_pop > 0 else 0.0

        self.prev_attack_gain_pred = gain
        self.prev_player_to_attack = player_to_attack
        self.prev_popularities_a_pred = popularities

    def n_attack_tokens(self, attack_tokens: np.array) -> None:
        n_attack = sum([tokens for tokens in attack_tokens])

        self.does_not_attack_too_much = 1 - (n_attack / self.n_tokens)

    # ------------------------------------------------------------------------------------------------------------------
    # Give tokens checkers ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def n_give_tokens(self, token_allocations: np.array) -> None:
        # Calculate how many of the CAB's tokens it plans to give
        n_give = 0

        for i in range(self.n_players):
            n_give += 1 if (i != self.player_idx and token_allocations[i] > 0) else 0

        self.does_not_give_too_much = 1 - (n_give / self.n_players)

    def tokens_to_desired_community(self, token_allocations: np.array, desired_community) -> None:
        # Calculate how many players in the CAB's desired community it plans to give tokens to
        desired_group = set(desired_community.s)
        n_give_to_desired_group = 0

        # The desired group is just us by ourselves
        if len(desired_group) == 1:
            n_give_to_desired_group = 1 if token_allocations[self.player_idx] > 0 else 0
            denominator = 1

        # Otherwise, there's at least one other player in the desired group
        else:
            for i in range(self.n_players):
                n_give_to_desired_group += 1 if (i != self.player_idx and i in desired_group) else 0
            denominator = len(desired_group) - 1

        self.gives_to_all_players_in_desired_group = n_give_to_desired_group / denominator

    def friends_reciprocate(self, token_allocations: np.array, influence_matrix: np.array) -> None:
        # Calculate, out of the players the CAB plans to give tokens to, how many reciprocated at least roughly the
        # same amount of popularity in the previous round
        friend_indices = list(np.where(token_allocations > 0)[0])
        n_friends, n_friends_who_reciprocate = 0, 0

        for friend_idx in friend_indices:
            if friend_idx == self.player_idx:
                continue
            n_friends += 1
            my_influence_on_friend = influence_matrix[self.player_idx][friend_idx]
            friends_influence_on_me = influence_matrix[friend_idx][self.player_idx]
            n_friends_who_reciprocate += 1 if friends_influence_on_me >= (my_influence_on_friend * 0.9) else 0

        self.all_friends_reciprocate = n_friends_who_reciprocate / n_friends if n_friends > 0 else 1.0

    def friends_that_have_attacked(self, token_allocations: np.array, influence_matrix: np.array) -> None:
        # Calculate, out of the players the CAB plans to give tokens to, how many have attacked other players in the
        # past
        for i in range(self.n_players):
            for j in range(self.n_players):
                if i == j:
                    continue

                i_influence_on_j = influence_matrix[i][j]

                if i_influence_on_j < 0:
                    players_i_has_attacked = self.players_that_have_attacked.get(i, [])
                    if j not in players_i_has_attacked:
                        players_i_has_attacked.append(j)
                    self.players_that_have_attacked[i] = players_i_has_attacked

        n_that_have_attacked, n_friends = 0, 0

        for i in range(self.n_players):
            n_that_have_attacked += 1 if (i != self.player_idx and token_allocations[
                i] > 0 and i in self.players_that_have_attacked) else 0
            n_friends += 1 if (i != self.player_idx and token_allocations[i] > 0) else 0

        self.no_friends_have_attacked = (1 - (n_that_have_attacked / n_friends)) if n_friends > 0 else (1 - float(
            self.player_idx in self.players_that_have_attacked))

        # Calculate, out of the players the CAB plans to give tokens to, how many have attacked us in the past
        n_that_have_attacked_us = 0

        for i, players_i_has_attacked in self.players_that_have_attacked.items():
            if i == self.player_idx:
                continue

            n_that_have_attacked_us += 1 if self.player_idx in players_i_has_attacked else 0

        self.no_friends_have_attacked_us = 1 - (n_that_have_attacked_us / n_friends) if n_friends > 0 else 1.0

    def give_results(self, tokens_that_were_given: np.array, desired_community, was_used: bool, round_num: int,
                     influence_matrix: np.array) -> None:
        desired_group = set(desired_community.s)

        if self.prev_tokens_that_were_given is not None:
            # Of the players that were actually given to, calculate how many are in the CAB's desired group
            n_in_desired_group = 0

            if len(desired_group) == 1:
                n_in_desired_group = 1 if self.prev_tokens_that_were_given[self.player_idx] > 0 else 0
                denominator = 1

            else:
                for i in range(self.n_players):
                    n_in_desired_group += 1 if (i != self.player_idx and i in desired_group) else 0
                denominator = len(desired_group) - 1

            self.given_to_all_in_desired_group = n_in_desired_group / denominator

            # Calculate how much friend reciprocation the CAB received within 2 rounds after the last time it was used
            if was_used:
                self.round_previously_used_g = round_num
                self.our_influence_last_time, self.their_influence_within_2 = {}, {}

                for i in range(self.n_players):
                    if i != self.player_idx and self.prev_tokens_that_were_given[self.player_idx] > 0:
                        self.our_influence_last_time[i] = influence_matrix[self.player_idx][i]
                        self.their_influence_within_2[i] = influence_matrix[i][self.player_idx]

            if self.round_previously_used_g is not None and round_num == self.round_previously_used_g + 1:
                for i in self.their_influence_within_2.keys():
                    self.their_influence_within_2[i] = max(self.their_influence_within_2[i],
                                                           influence_matrix[i][self.player_idx])

            n_friends, n_that_reciprocated = len(self.our_influence_last_time), 0

            for i, our_influence in self.our_influence_last_time.items():
                their_influence = self.their_influence_within_2[i]
                n_that_reciprocated += 1 if their_influence >= (our_influence * 0.9) else 0

            self.all_friends_reciprocated_within_2_last_time = n_that_reciprocated / n_friends if n_friends > 0 else 1.0

        self.prev_tokens_that_were_given = tokens_that_were_given

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def assumptions(self) -> Assumptions:
        pass
