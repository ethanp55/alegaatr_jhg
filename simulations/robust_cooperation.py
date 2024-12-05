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


class KillWeakestThieves(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.whoami = 'kill_weakest_thieves'
        self.gameParams = {}
        self.attackers = set()

    def setGameParams(self, gameParams, visualTraits):
        self.gameParams = gameParams

    def play_round(self, player_idx, round_num, recieved, popularities, influence, extra_data, v, transactions):
        if v is not None:
            for i in range(len(v)):
                for j in range(len(v[0])):
                    if v[i][j] < 0 and i != player_idx:
                        self.attackers.add(i)
            weakest_i, weakest_pop = -1, np.inf
            for i in self.attackers:
                pop = popularities[i]
                if pop < weakest_pop:
                    weakest_i, weakest_pop = i, pop
            assert weakest_i != player_idx
            tkns = [0] * len(popularities)
            if weakest_i != -1 and weakest_pop > 0:
                tkns[weakest_i] = -len(popularities) * 2
            else:
                n_friends = len(popularities) - len(self.attackers)
                assert n_friends >= 1
                n_tokens_per_friend = (len(popularities) * 2) // n_friends
                for i in range(len(popularities)):
                    if i not in self.attackers:
                        tkns[i] = n_tokens_per_friend

        else:
            tkns = [2] * len(popularities)

        return np.array(tkns)


class Coop(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.whoami = 'coop'
        self.gameParams = {}

    def setGameParams(self, gameParams, visualTraits):
        self.gameParams = gameParams

    def play_round(self, player_idx, round_num, recieved, popularities, influence, extra_data, v, transactions):
        tkns = [2] * len(popularities)

        return np.array(tkns)


N_EPOCHS = 5
keep_all = GeneAgent3('all_keep', 1)
equal_steal = EqualSteal()
kill_weakest_thieves = KillWeakestThieves()
assassin = GeneAgent3(
    'gene_0_0_1_25_0_50_100_0_0_0_0_100_0_50_50_0_0_100_10_90_0_0_50_100_100_100_100_100_80_100_100_0_0', 1)
coop = Coop()
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
        list_of_opponents.append(([deepcopy(keep_all) for _ in range(n_players - 1)], 'keep_all'))
        list_of_opponents.append(([deepcopy(equal_steal) for _ in range(n_players - 1)], 'equal_steal'))
        list_of_opponents.append(([deepcopy(kill_weakest_thieves) for _ in range(n_players - 2)] + [deepcopy(assassin)],
                                  'self_play_assassin'))
        list_of_opponents.append(([deepcopy(coop) for _ in range(n_players - 1)], 'coop'))

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
                opps = deepcopy(opponents)
                players = create_society(agent_to_test, [], opps, n_players)
                pops_file = f'../simulations/robust_coop_scores/{agent_to_test.whoami}_{opponents_label}.csv'
                run_with_specified_agents(players=players, initial_pop_setting='equal', numRounds=n_rounds,
                                          final_pops_file=pops_file)


if __name__ == '__main__':
    robust_coop()
