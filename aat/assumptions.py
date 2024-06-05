from dataclasses import dataclass
from typing import List


# CAB assumption checkers:
#   - Detect communities (6/10):
#       - Have there been recent significant changes in community memberships?
#       - Are the communities and/or modularity similar when IHN is calculated with a min instead of a max?
#       - Are the communities and/or modularity similar when IHP is calculated with a min instead of a max?
#
#   - Determine desired community (these pertain to the specific community chosen by the CAB agent) (6/11 - 6/12):
#       - 2.1 - Have there been recent significant changes in collective community popularity?
#       - 2.2 - Have there been recent significant changes in closeness to target group strength?
#       - 3 - Have there been recent significant changes in prominence?
#       - 4 - Have there been recent significant changes in familiarity scores?
#       - 5 - Have there been recent significant changes in prosocial behavior?
#
#   - Determine number of tokens to keep (6/13):
#       - Are the attack predictions accurate?
#
#   - Determine who to attack (6/13):
#       - Does the agent receive profit from attacking player i?
#       - Does player i receive damage?
#
#   - Give tokens to members of chosen community (6/14):
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

    def alignment_vector(self) -> List[float]:
        attribute_names = self.__dict__.keys()
        tup = [self.__getattribute__(name) for name in attribute_names]

        return tup
