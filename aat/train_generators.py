# TODO: ask Dr. Crandall about both of these points
# How to create training labels:
#   - Average popularity increase per round, from the end of the current round to the end of the game
#   - Baseline = 25

# General training conditions:
#   - 5 different initial popularities
#   - 5 players, 10 players, 15 players, 20 players
#   - 20 rounds, 30 rounds, 40 rounds, 50 rounds, 100 rounds
#   - 0 cats, 1 cat, 2 cats
#   - Multiple epochs (maybe 30?)

# Opponents:
#   - Random selection of generators
#   - CABs with randomly-selected parameters
#   - Random selection of best CABs when trained with no cats
#   - Random selection of best CABs when trained with 1 cat
#   - Random selection of best CABs when trained with 2 cats
#   - Random mixture of all of the above

# Generator training conditions:
#   - Basic bandit with epsilon = 0.05, decay = 0.99
#   - Basic bandit with epsilon = 0.1, decay = 0.99
#   - Basic bandit with epsilon = 0.2, decay = 0.99
#   - Agent that randomly selects generators - uniform
#   - Agent that randomly selects generators - based on how long it's been since last used (more recent is more likely)
#   - BBL
#   - S++
