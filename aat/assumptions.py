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
    improved_from_prev_round: float
    improved_medium_term: float
    improved_long_term: float
    rank: float
    rank_improved_from_prev_round: float
    rank_improved_medium_term: float
    rank_improved_long_term: float
    percentile: float
    below_30_rounds: float
    above_30_rounds: float
    below_10_players: float
    above_10_players: float
    positive_density: float
    negative_density: float
    percentage_of_positive_edges: float
    percentage_of_neutral_edges: float
    percentage_of_negative_edges: float
    modularity_above_ema: float
    modularity_below_ema: float
    communities_changes_from_prev: float
    communities_diffs_with_ihn_max: float
    communities_diffs_with_ihp_min: float
    below_prev_collective_strength: float
    above_prev_collective_strength: float
    below_target_strength: float
    above_target_strength: float
    percent_of_players_needed_for_desired_community: float
    prominence_below_avg: float
    prominence_above_avg: float
    prominence_max_val: float
    prominence_rank_val: float
    familiarity_below_modularity: float
    familiarity_above_modularity: float
    prosocial_score: float
    percent_tokens_kept: float
    percent_attackers: float
    percent_pop_of_attackers: float
    percent_impact_of_attackers: float
    tokens_kept_below_stolen: float
    tokens_kept_above_stolen: float
    my_attack_damaged_other_player: float
    my_attack_benefited_me: float
    vengence_attack: float
    defend_friend_attack: float
    pillage_attack: float
    percent_of_players_to_give_to: float
    percent_of_friends_who_reciprocate: float

    def __post_init__(self) -> None:
        attribute_vals = self.__dict__.values()

        for val in attribute_vals:
            assert 0 <= val <= 1

    def alignment_vector(self) -> List[float]:
        attribute_vals = self.__dict__.values()
        tup = [round(val, 5) for val in attribute_vals]

        return tup
