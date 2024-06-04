from copy import deepcopy
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.main import run_with_specified_agents
import pandas as pd
from scipy.stats import rankdata

MAX_PLAYERS = 10

generators, generator_df = [], pd.read_csv(f'../ResultsSaved/generator_genes/genes.csv', header=None)

for i in range(len(generator_df)):
    gene_str = generator_df.iloc[i, 0]
    generators.append(GeneAgent3(gene_str, 1))

# Hand-coded CAB agent designed to work well against cats
# hand_coded = GeneAgent3('assassin', 1)
# generators.append(hand_coded)
#
# assert len(generators) == 17
assert len(generators) == 16

varied_cabs_1 = [GeneAgent3('', 1) for _ in range(MAX_PLAYERS)]
varied_cabs_2 = [GeneAgent3('', 1) for _ in range(MAX_PLAYERS)]
varied_cabs_3 = [GeneAgent3('', 1) for _ in range(MAX_PLAYERS)]
varied_cabs_4 = [GeneAgent3('', 1) for _ in range(MAX_PLAYERS)]
varied_cabs_5 = [GeneAgent3('', 1) for _ in range(MAX_PLAYERS)]

same_cab_1, same_cab_2, same_cab_3, same_cab_4, same_cab_5 = \
    GeneAgent3('', 1), GeneAgent3('', 1), GeneAgent3('', 1), GeneAgent3('', 1), GeneAgent3('', 1)
same_cabs_1 = [deepcopy(same_cab_1) for _ in range(MAX_PLAYERS)]
same_cabs_2 = [deepcopy(same_cab_2) for _ in range(MAX_PLAYERS)]
same_cabs_3 = [deepcopy(same_cab_3) for _ in range(MAX_PLAYERS)]
same_cabs_4 = [deepcopy(same_cab_4) for _ in range(MAX_PLAYERS)]
same_cabs_5 = [deepcopy(same_cab_5) for _ in range(MAX_PLAYERS)]

assassins = [AssassinAgent(), AssassinAgent()]

test_results, test_ranks = {}, {}

for i, generator in enumerate(generators):
    print(f'Testing generator {i}')

    # Test condition 1
    players = [generator] + varied_cabs_1[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_1'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_1', False)
    test_ranks['varied_1'] = max(generator_rank, test_ranks.get('varied_1', 0))

    # Test condition 2
    players = [generator] + varied_cabs_2[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_2'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_2', False)
    test_ranks['varied_2'] = max(generator_rank, test_ranks.get('varied_2', 0))

    # Test condition 3
    players = [generator] + varied_cabs_3[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_3'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_3', False)
    test_ranks['varied_3'] = max(generator_rank, test_ranks.get('varied_3', 0))

    # Test condition 4
    players = [generator] + same_cabs_1[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_1'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_1', False)
    test_ranks['same_1'] = max(generator_rank, test_ranks.get('same_1', 0))

    # Test condition 5
    players = [generator] + same_cabs_2[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_2'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_2', False)
    test_ranks['same_2'] = max(generator_rank, test_ranks.get('same_2', 0))

    # Test condition 6
    players = [generator] + same_cabs_3[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_3'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_3', False)
    test_ranks['same_3'] = max(generator_rank, test_ranks.get('same_3', 0))

    # Test condition 7
    players = [generator] + varied_cabs_4[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_4_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_4_1cat', False)
    test_ranks['varied_4_1cat'] = max(generator_rank, test_ranks.get('varied_4_1cat', 0))

    # Test condition 8
    players = [generator] + varied_cabs_5[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_5_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_5_1cat', False)
    test_ranks['varied_5_1cat'] = max(generator_rank, test_ranks.get('varied_5_1cat', 0))

    # Test condition 9
    players = [generator] + same_cabs_4[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_4_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_4_1cat', False)
    test_ranks['same_4_1cat'] = max(generator_rank, test_ranks.get('same_4_1cat', 0))

    # Test condition 10
    players = [generator] + same_cabs_5[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_5_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_5_1cat', False)
    test_ranks['same_5_1cat'] = max(generator_rank, test_ranks.get('same_5_1cat', 0))

    # Test condition 11
    players = [generator] + varied_cabs_4[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_4_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_4_2cats', False)
    test_ranks['varied_4_2cats'] = max(generator_rank, test_ranks.get('varied_4_2cats', 0))

    # Test condition 12
    players = [generator] + varied_cabs_5[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_5_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_5_2cats', False)
    test_ranks['varied_5_2cats'] = max(generator_rank, test_ranks.get('varied_5_2cats', 0))

    # Test condition 13
    players = [generator] + same_cabs_4[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_4_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_4_2cats', False)
    test_ranks['same_4_2cats'] = max(generator_rank, test_ranks.get('same_4_2cats', 0))

    # Test condition 14
    players = [generator] + same_cabs_5[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_5_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_5_2cats', False)
    test_ranks['same_5_2cats'] = max(generator_rank, test_ranks.get('same_5_2cats', 0))

print(test_results)
print(test_ranks)
