from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.main import *

if __name__ == '__main__':
    if len(sys.argv) < 10:
        print('Not enough arguments')
        sys.exit(1)

    # num players
    # num rounds
    # num alegaatrs
    # num cats
    # initial pop condition
    # opponents:
    #   - equal distribution of bandits
    #   - random selection of bandits
    #   - random selection of generators
    #   - cabs with random params
    #   - random selection of best cabs
    #   - random mixture of all of the above
