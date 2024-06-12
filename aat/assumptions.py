from dataclasses import dataclass
from typing import List


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

    def alignment_vector(self) -> List[float]:
        attribute_names = self.__dict__.keys()
        tup = [round(self.__getattribute__(name), 5) for name in attribute_names]

        return tup
