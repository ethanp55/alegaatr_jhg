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


class Assumptions:
    def __init__(self) -> None:
        pass
