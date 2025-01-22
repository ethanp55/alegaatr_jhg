from copy import deepcopy
from GeneSimulation_py.alegaatr import AlegAATr
from GeneSimulation_py.aleqgaatr import AleqgAATr
from GeneSimulation_py.baseagent import AbstractAgent
from GeneSimulation_py.dqn import DQNAgent
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.generator_pool import GeneratorPool
from GeneSimulation_py.madqn import MADQN
from GeneSimulation_py.main import run_with_specified_agents
from GeneSimulation_py.qalegaatr import QAlegAATr
from GeneSimulation_py.raat import RAAT
from GeneSimulation_py.ralegaatr import RAlegAATr
from GeneSimulation_py.rawo import RawO
from GeneSimulation_py.rdqn import RDQN
from GeneSimulation_py.smalegaatr import SMAlegAATr
from GeneSimulation_py.soaleqgaatr import SOAleqgAATr
import numpy as np
import os
from GeneSimulation_py.tftagent import TFTAgent


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


N_EPOCHS = 30
keep_all = GeneAgent3('all_keep', 1)
equal_steal = EqualSteal()
n_rounds, n_players = 30, 10

tft_gene_strings = [
    'tft_16_0_0_57_80_11_57',
    'tft_10_0_0_98_83_2_69',
    'tft_56_4_0_82_91_47_32',
    'tft_4_1_2_82_29_4_57',
    'tft_4_2_0_57_80_9_18',
    'tft_15_86_5_100_88_76_18',
    'tft_40_47_0_53_77_16_27',
    'tft_16_0_0_79_80_11_23',
    'tft_38_6_0_90_77_5_18',
    'tft_46_51_0_7_72_63_57',
]
assert len(tft_gene_strings) == n_players


# names = ['DQN', 'MADQN', 'RDQN', 'AleqgAATr', 'RAlegAATr', 'SOAleqgAATr', 'AlegAATr', 'SMAlegAATr', 'QAlegAATr', 'RawO',
#          'PPO', 'RAAT']


def adaptability(epoch_number) -> None:
    # # Reset any existing simulation files (opening a file in write mode will truncate it)
    # for file in os.listdir('../simulations/adaptability_results/'):
    #     if 'coop' in file:
    #         continue
    #     name = file.split('_')[0]
    #     if name in names:
    #         with open(f'../simulations/adaptability_results/{file}', 'w', newline='') as _:
    #             pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch + 1}')
        list_of_opponents = []
        list_of_opponents.append(([deepcopy(keep_all) for _ in range(n_players - 1)], 'keep_all'))
        list_of_opponents.append(([deepcopy(equal_steal) for _ in range(n_players - 1)], 'equal_steal'))
        list_of_opponents.append(([], 'selfplay'))
        list_of_opponents.append(([], 'coop1'))
        list_of_opponents.append(([], 'coop2'))

        for opponents, opponents_label in list_of_opponents:
            agents_to_test = []
            agents_to_test.append(AlegAATr(lmbda=0.0, ml_model_type='knn', enhanced=True))
            agents_to_test.append(RAlegAATr(train_network=False))
            agents_to_test.append(AleqgAATr(train_network=False))
            # agents_to_test.append(SMAlegAATr(enhanced=True))
            agents_to_test.append(DQNAgent(train_network=False))
            # agents_to_test.append(SOAleqgAATr(train_network=False))
            agents_to_test.append(QAlegAATr(enhanced=True))
            agents_to_test.append(RawO(enhanced=True))
            agents_to_test.append(RAAT(enhanced=True))
            # agents_to_test.append(RDQN(train_network=False))
            # agents_to_test.append(MADQN(train_networks=False))

            for agent_to_test in agents_to_test:
                if opponents_label == 'selfplay':
                    opps = [deepcopy(agent_to_test) for _ in range(n_players - 1)]

                elif opponents_label == 'coop1':
                    opps = [TFTAgent(geneStr=gene_str) for gene_str in tft_gene_strings[:5]]
                    opps += [deepcopy(agent_to_test) for _ in range(4)]

                elif opponents_label == 'coop2':
                    opps = [TFTAgent(geneStr=gene_str) for gene_str in tft_gene_strings[:9]]

                else:
                    opps = deepcopy(opponents)

                players = opps + [agent_to_test]
                assert len(players) == n_players
                pops_file = f'../simulations/adaptability_results/{agent_to_test.whoami}_{opponents_label}_epoch={epoch_number}.csv'
                run_with_specified_agents(players=players, initial_pop_setting='equal', numRounds=n_rounds,
                                          final_pops_file=pops_file)


# def coop() -> None:
#     # Reset any existing simulation files (opening a file in write mode will truncate it)
#     for file in os.listdir('../simulations/adaptability_results/'):
#         if 'coop' not in file:
#             continue
#         name = file.split('_')[0]
#         if name in names:
#             with open(f'../simulations/adaptability_results/{file}', 'w', newline='') as _:
#                 pass
#
#     for epoch in range(N_EPOCHS):
#         print(f'Epoch: {epoch + 1}')
#         cooperators = [AleqgAATr(train_network=False),
#                        RawO(enhanced=True),
#                        QAlegAATr(enhanced=True),
#                        AlegAATr(lmbda=0.0, ml_model_type='knn', enhanced=True)]
#         cooperators_types = [type(cooperator) for cooperator in cooperators]
#         cooperator_indices = np.arange(len(cooperators))
#
#         agents_to_test = []
#         agents_to_test.append(AlegAATr(lmbda=0.0, ml_model_type='knn', enhanced=True))
#         agents_to_test.append(RAlegAATr(train_network=False))
#         agents_to_test.append(AleqgAATr(train_network=False))
#         agents_to_test.append(SMAlegAATr(enhanced=True))
#         agents_to_test.append(DQNAgent(train_network=False))
#         agents_to_test.append(SOAleqgAATr(train_network=False))
#         agents_to_test.append(QAlegAATr(enhanced=True))
#         agents_to_test.append(RawO(enhanced=True))
#         agents_to_test.append(RAAT(enhanced=True))
#         agents_to_test.append(RDQN(train_network=False))
#         agents_to_test.append(MADQN(train_networks=False))
#
#         for agent_to_test in agents_to_test:
#             print(agent_to_test.whoami)
#             if type(agent_to_test) in cooperators_types:
#                 idx_to_exclude = cooperators_types.index(type(agent_to_test))
#                 filtered_indices = cooperator_indices[cooperator_indices != idx_to_exclude]
#                 opp_indices = np.random.choice(filtered_indices, n_players)
#             else:
#                 opp_indices = np.random.choice(len(cooperators), n_players)
#             opps = [deepcopy(cooperators[idx]) for idx in opp_indices]
#             players = create_society(agent_to_test, [], opps, n_players)
#             pops_file = f'../simulations/adaptability_results/{agent_to_test.whoami}_coop.csv'
#             run_with_specified_agents(players=players, initial_pop_setting='equal', numRounds=n_rounds,
#                                       final_pops_file=pops_file)


def baselines() -> None:
    results = {}

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch + 1}')

        list_of_opponents = []
        list_of_opponents.append(([deepcopy(keep_all) for _ in range(n_players - 1)], 'keep_all'))
        list_of_opponents.append(([deepcopy(equal_steal) for _ in range(n_players - 1)], 'equal_steal'))
        list_of_opponents.append(([], 'selfplay'))
        # list_of_opponents.append(([], 'coop1'))
        # list_of_opponents.append(([], 'coop2'))

        for opponents, opponents_label in list_of_opponents:
            agents_to_test = GeneratorPool().generators

            for generator_idx, agent_to_test in enumerate(agents_to_test):
                if generator_idx not in results:
                    results[generator_idx] = {}

                if opponents_label == 'selfplay':
                    opps = [deepcopy(agent_to_test) for _ in range(n_players - 1)]

                elif opponents_label == 'coop1':
                    opps = [TFTAgent(geneStr=gene_str) for gene_str in tft_gene_strings[:5]]
                    opps += [deepcopy(agent_to_test) for _ in range(4)]

                elif opponents_label == 'coop2':
                    opps = [TFTAgent(geneStr=gene_str) for gene_str in tft_gene_strings[:9]]

                else:
                    opps = deepcopy(opponents)

                players = opps + [agent_to_test]
                assert len(players) == n_players
                final_pops = run_with_specified_agents(players=players, initial_pop_setting='equal',
                                                       numRounds=n_rounds)
                results[generator_idx][opponents_label] = results[generator_idx].get(opponents_label, []) + \
                                                          [final_pops[-1]]

    for generator_idx, res in results.items():
        print(generator_idx)
        for opp_type, pops in res.items():
            print(f'{opp_type}: {pops}, avg = {np.array(pops).mean()}')
        print()


if __name__ == '__main__':
    baselines()
