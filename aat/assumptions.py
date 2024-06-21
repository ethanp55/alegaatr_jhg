from dataclasses import dataclass
from typing import List


# Progress checkers:
#   - Relative popularity
#       - Could have a few assumptions around something like percentiles or even quartiles (25th, 50th, 90th, etc.)
#       - Could have an assumption about whether our position is better than previous round (short-term)
#       - Could have an assumption about whether our position is better than ~5 rounds ago (medium-term)
#       - Could have an assumption about whether our position is better than ~10 rounds ago (long-term)
#   - Growing more popular:
#       - Did popularity increase since the previous round (short-term)?
#       - Maybe something like whether average popularity is larger than the average popularity from ~5 rounds ago
#           (medium-term)
#       - Maybe something like whether average popularity is larger than the average popularity from ~10 rounds ago
#           (long-term)
#   - Rounds:
#       - How many rounds above/below 30 are we (since the GA uses 30 rounds)?
#   - Population:
#       - How many players above/below 10 are there (since the GA uses 10 agents)?

# CAB assumption checkers:
#   - Detect communities:
#       - Is the graph reasonably connected (not sure if this important or useful)?
#       - Has modularity significantly changed recently?
#       - Has the distribution of edge weights (both positive and negative) significantly changed recently?
#       - Have there been recent significant changes in community memberships?
#       - Are the communities and/or modularity similar when IHN is calculated with a min instead of a max?
#       - Are the communities and/or modularity similar when IHP is calculated with a min instead of a max?
#
#   - Determine desired community (these pertain to the specific community chosen by the CAB agent):
#       - 1 - Have there been recent significant changes in modularity?
#       - 2.1 - Have there been recent significant changes in collective community popularity?
#       - 2.2 - Have there been recent significant changes in closeness to target group strength?
#       - 3 - Have there been recent significant changes in prominence?
#       - 4 - Have there been recent significant changes in familiarity scores?
#       - 5 - Have there been recent significant changes in prosocial behavior?
#
#   - Determine number of tokens to keep:
#       - Are the attack predictions accurate?
#
#   - Determine who to attack:
#       - Does the agent receive profit from attacking player i?
#       - Does player i receive damage?
#
#   - Give tokens to members of chosen community:
#       - Are my friends reciprocating an equal or greater amount of popularity?

# Assumptions about other players:
#   - Not sure if we need these, because the CAB agent checkers might tease most/all of these out

# Initial popularity checkers (only needed if we decided to use different conditions in our experiments - also want to
# double-check if even having the knowledge of varying initial popularities and how they're calculated is cheating):
#   - Check if everyone has the same starting popularity (easy)
#   - Check if initial popularity is random
#   - Check if initial popularity is high-low
#   - Check if initial popularity is step
#   - Otherwise, initial popularity is "power"


@dataclass
class Assumptions:
    # --------------------------------------------------------------------------------------------------------------
    # Assumption estimates from progress checkers ------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    pop_improved_1_round_after: float
    pop_improved_2_rounds_after: float
    rel_pop_improved_1_round_after: float
    rel_pop_improved_2_rounds_after: float
    rank_1_round_after: float
    rank_2_rounds_after: float
    curr_rel_pop: float
    curr_rank: float
    below_30_rounds: float
    above_30_rounds: float
    below_10_players: float
    above_10_players: float
    was_just_used: float
    positive_density: float
    negative_density: float
    modularity_above_ema: float
    modularity_below_ema: float
    communities_changes_from_prev: float
    communities_diffs_with_ihn_max: float
    communities_diffs_with_ihp_min: float
    communities_diffs_with_just_used: float
    communities_diffs_from_last_use: float
    collective_strength_increased: float
    community_has_significant_strength: float
    near_target_strength: float
    percent_of_players_needed_for_desired_community: float
    prominence_avg_val: float
    prominence_max_val: float
    prominence_rank_val: float
    familiarity_better_than_modularity: float
    prosocial_score: float
    desired_comm_diffs_with_just_used: float
    desired_comm_diffs_from_last_use: float
    does_not_keep_too_much: float
    n_attackers_is_low: float
    attackers_are_weak: float
    defense_was_effective: float
    defense_was_effective_last_time: float
    defense_would_have_been_effective: float
    none_in_desired_community: float
    none_in_existing_community: float
    my_last_attack_damaged_other_player: float
    my_last_attack_benefited_me: float
    pop_did_not_decrease_after_last_attack: float
    attack_would_have_damaged_other: float
    attack_would_have_benefited_us: float
    does_not_attack_too_much: float
    attacked_player_not_in_community: float
    attacked_player_not_in_desired_group: float
    attack_damaged_other_player: float
    attack_benefited_me: float
    does_not_give_too_much: float
    gives_to_multiple_players: float
    gives_to_all_players_in_desired_group: float
    all_friends_reciprocate: float
    no_friends_have_attacked: float
    no_friends_have_attacked_us: float
    given_to_all_in_desired_group: float
    all_friends_reciprocated_within_2_last_time: float

    def __post_init__(self) -> None:
        # for name, val in self.__dict__.items():
        #     if not 0 <= val <= 1:
        #         print('ISSUE WITH ASSUMPTION VALUE: ', name, val)

        for val in self.__dict__.values():
            assert 0 <= val <= 1

    def alignment_vector(self) -> List[float]:
        attribute_vals = self.__dict__.values()
        tup = [round(val, 5) for val in attribute_vals]

        return tup
