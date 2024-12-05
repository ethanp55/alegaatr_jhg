from aat.train_generators import FavorMoreRecent, UniformSelector
from aat.train_generators import cabs_with_random_params, random_selection_of_best_trained_cabs, random_agents, \
    basic_bandits, random_mixture_of_all_types, create_society
from copy import deepcopy
from GeneSimulation_py.aalegaatr import AAlegAATr
from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.aleqgaatr import AleqgAATr
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.dqn import DQNAgent
from GeneSimulation_py.ducb import DUCB
from GeneSimulation_py.eee import EEE
from GeneSimulation_py.exp4 import EXP4
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.madqn import MADQN
from GeneSimulation_py.main import run_with_specified_agents
from GeneSimulation_py.qalegaatr import QAlegAATr
from GeneSimulation_py.ralegaatr import RAlegAATr
from GeneSimulation_py.rawo import RawO
from GeneSimulation_py.rdqn import RDQN
from GeneSimulation_py.rucb import RUCB
from GeneSimulation_py.smalegaatr import SMAlegAATr
from GeneSimulation_py.soaleqgaatr import SOAleqgAATr
from GeneSimulation_py.swucb import SWUCB
from GeneSimulation_py.ucb import UCB
import numpy as np
import os
import pandas as pd
from typing import List


class EqualSteal(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.whoami = 'equal_steal'
        self.gameParams = {}

    def setGameParams(self, gameParams, visualTraits):
        self.gameParams = gameParams

    def play_round(self, player_idx, round_num, recieved, popularities, influence, extra_data, v, transactions):
        tkns = [-2] * len(popularities)
        tkns[player_idx] = abs(tkns[player_idx])

        return np.array(tkns)


N_EPOCHS = 5
keep_all = GeneAgent3('all_keep', 1)
equal_steal = EqualSteal()
n_rounds, n_players = 20, 15
names = []


def robust_coop() -> None:
    # Reset any existing simulation files (opening a file in write mode will truncate it)
    for file in os.listdir('../simulations/robust_coop_scores/'):
        name = file.split('_')[0]
        if name in names:
            with open(f'../simulations/robust_coop_scores/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch + 1}')
        list_of_opponents = []
        # list_of_opponents.append(([deepcopy(keep_all) for _ in range(n_players - 1)], 'keep_all'))
        # list_of_opponents.append(([deepcopy(equal_steal) for _ in range(n_players - 1)], 'equal_steal'))
        list_of_opponents.append(([], 'coop'))

        for opponents, opponents_label in list_of_opponents:
            agents_to_test = []
            agents_to_test.append(AlegAATr(lmbda=0.0, ml_model_type='knn', enhanced=True))
            agents_to_test.append(RAlegAATr(train_network=False))
            agents_to_test.append(AleqgAATr(train_network=False))
            agents_to_test.append(SMAlegAATr(enhanced=True))
            agents_to_test.append(DQNAgent(train_network=False))
            agents_to_test.append(SOAleqgAATr(train_network=False))
            agents_to_test.append(QAlegAATr(enhanced=True))
            agents_to_test.append(RawO(enhanced=True))

            for agent_to_test in agents_to_test:
                opps = [deepcopy(agent_to_test) for _ in range(n_players - 1)] if opponents_label == 'coop' else \
                    deepcopy(opponents)
                players = create_society(agent_to_test, [], opps, n_players)
                pops_file = f'../simulations/robust_coop_scores/{agent_to_test.whoami}_{opponents_label}.csv'
                run_with_specified_agents(players=players, initial_pop_setting='equal', numRounds=n_rounds,
                                          final_pops_file=pops_file)


if __name__ == '__main__':
    robust_coop()
