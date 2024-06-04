from copy import deepcopy
from GeneSimulation_py.assassinagent import AssassinAgent
from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.main import run_with_specified_agents
import os
import pandas as pd
from scipy.stats import rankdata

MAX_PLAYERS = 10
TRAINED_CAB_GENERATIONS = ['0', '49', '99', '149', '199']

generators, generator_df = [], pd.read_csv(f'../ResultsSaved/generator_genes/genes.csv', header=None)

for i in range(0, len(generator_df), 2):
    gene_str = generator_df.iloc[i, 0]
    generators.append(GeneAgent3(gene_str, 1))

no_cat_files = os.listdir('../ResultsSaved/no_cat/')
no_cat_files = [file for file in no_cat_files if
                file.split('_')[1].split('.')[0] in ['24', '49', '74', '99', '124', '149', '174', '199']]
for file in no_cat_files:
    gen_num = file.split('_')[1].split('.')[0]
    assert gen_num in ['24', '49', '74', '99', '124', '149', '174', '199']
    df = pd.read_csv(f'../ResultsSaved/no_cat/{file}', header=None)
    gene_str = df.iloc[0, 0]
    cab = GeneAgent3(gene_str, 1)
    generators.append(cab)

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

trained_no_cat_gen0_cabs, trained_no_cat_gen49_cabs, trained_no_cat_gen99_cabs, trained_no_cat_gen149_cabs, \
trained_no_cat_gen199_cabs = [], [], [], [], []
no_cat_files = os.listdir('../ResultsSaved/no_cat/')
no_cat_files = [file for file in no_cat_files if file.split('_')[1].split('.')[0] in TRAINED_CAB_GENERATIONS]
for file in no_cat_files:
    gen_num = file.split('_')[1].split('.')[0]
    assert gen_num in TRAINED_CAB_GENERATIONS
    df = pd.read_csv(f'../ResultsSaved/no_cat/{file}', header=None)

    for i in range(MAX_PLAYERS):
        gene_str = df.iloc[i, 0]
        cab = GeneAgent3(gene_str, 1)

        if gen_num == '0':
            trained_no_cat_gen0_cabs.append(cab)

        elif gen_num == '49':
            trained_no_cat_gen49_cabs.append(cab)

        elif gen_num == '99':
            trained_no_cat_gen99_cabs.append(cab)

        elif gen_num == '149':
            trained_no_cat_gen149_cabs.append(cab)

        else:
            trained_no_cat_gen199_cabs.append(cab)
assert len(trained_no_cat_gen0_cabs) == len(trained_no_cat_gen49_cabs) == len(trained_no_cat_gen99_cabs) == \
       len(trained_no_cat_gen149_cabs) == len(trained_no_cat_gen199_cabs) == MAX_PLAYERS

trained_one_cat_gen0_cabs, trained_one_cat_gen49_cabs, trained_one_cat_gen99_cabs, trained_one_cat_gen149_cabs, \
trained_one_cat_gen199_cabs = [], [], [], [], []
no_cat_files = os.listdir('../ResultsSaved/one_cat/')
no_cat_files = [file for file in no_cat_files if file.split('_')[1].split('.')[0] in TRAINED_CAB_GENERATIONS]
for file in no_cat_files:
    gen_num = file.split('_')[1].split('.')[0]
    assert gen_num in TRAINED_CAB_GENERATIONS
    df = pd.read_csv(f'../ResultsSaved/one_cat/{file}', header=None)

    for i in range(MAX_PLAYERS):
        gene_str = df.iloc[i, 0]
        cab = GeneAgent3(gene_str, 1)

        if gen_num == '0':
            trained_one_cat_gen0_cabs.append(cab)

        elif gen_num == '49':
            trained_one_cat_gen49_cabs.append(cab)

        elif gen_num == '99':
            trained_one_cat_gen99_cabs.append(cab)

        elif gen_num == '149':
            trained_one_cat_gen149_cabs.append(cab)

        else:
            trained_one_cat_gen199_cabs.append(cab)
assert len(trained_one_cat_gen0_cabs) == len(trained_one_cat_gen49_cabs) == len(trained_one_cat_gen99_cabs) == \
       len(trained_one_cat_gen149_cabs) == len(trained_one_cat_gen199_cabs) == MAX_PLAYERS

trained_two_cat_gen0_cabs, trained_two_cat_gen49_cabs, trained_two_cat_gen99_cabs, trained_two_cat_gen149_cabs, \
trained_two_cat_gen199_cabs = [], [], [], [], []
no_cat_files = os.listdir('../ResultsSaved/two_cats/')
no_cat_files = [file for file in no_cat_files if file.split('_')[1].split('.')[0] in TRAINED_CAB_GENERATIONS]
for file in no_cat_files:
    gen_num = file.split('_')[1].split('.')[0]
    assert gen_num in TRAINED_CAB_GENERATIONS
    df = pd.read_csv(f'../ResultsSaved/two_cats/{file}', header=None)

    for i in range(MAX_PLAYERS):
        gene_str = df.iloc[i, 0]
        cab = GeneAgent3(gene_str, 1)

        if gen_num == '0':
            trained_two_cat_gen0_cabs.append(cab)

        elif gen_num == '49':
            trained_two_cat_gen49_cabs.append(cab)

        elif gen_num == '99':
            trained_two_cat_gen99_cabs.append(cab)

        elif gen_num == '149':
            trained_two_cat_gen149_cabs.append(cab)

        else:
            trained_two_cat_gen199_cabs.append(cab)
assert len(trained_two_cat_gen0_cabs) == len(trained_two_cat_gen49_cabs) == len(trained_two_cat_gen99_cabs) == \
       len(trained_two_cat_gen149_cabs) == len(trained_two_cat_gen199_cabs) == MAX_PLAYERS

test_results, test_ranks, test_popularities = {}, {}, {}

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
    test_popularities['varied_1'] = max(generator_pop, test_popularities.get('varied_1', 0))

    # Test condition 2
    players = [generator] + varied_cabs_2[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_2'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_2', False)
    test_ranks['varied_2'] = max(generator_rank, test_ranks.get('varied_2', 0))
    test_popularities['varied_2'] = max(generator_pop, test_popularities.get('varied_2', 0))

    # Test condition 3
    players = [generator] + varied_cabs_3[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_3'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_3', False)
    test_ranks['varied_3'] = max(generator_rank, test_ranks.get('varied_3', 0))
    test_popularities['varied_3'] = max(generator_pop, test_popularities.get('varied_3', 0))

    # Test condition 4
    players = [generator] + same_cabs_1[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_1'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_1', False)
    test_ranks['same_1'] = max(generator_rank, test_ranks.get('same_1', 0))
    test_popularities['same_1'] = max(generator_pop, test_popularities.get('same_1', 0))

    # Test condition 5
    players = [generator] + same_cabs_2[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_2'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_2', False)
    test_ranks['same_2'] = max(generator_rank, test_ranks.get('same_2', 0))
    test_popularities['same_2'] = max(generator_pop, test_popularities.get('same_2', 0))

    # Test condition 6
    players = [generator] + same_cabs_3[1:]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_3'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_3', False)
    test_ranks['same_3'] = max(generator_rank, test_ranks.get('same_3', 0))
    test_popularities['same_3'] = max(generator_pop, test_popularities.get('same_3', 0))

    # Test condition 7
    players = [generator] + varied_cabs_4[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_4_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_4_1cat', False)
    test_ranks['varied_4_1cat'] = max(generator_rank, test_ranks.get('varied_4_1cat', 0))
    test_popularities['varied_4_1cat'] = max(generator_pop, test_popularities.get('varied_4_1cat', 0))

    # Test condition 8
    players = [generator] + varied_cabs_5[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_5_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_5_1cat', False)
    test_ranks['varied_5_1cat'] = max(generator_rank, test_ranks.get('varied_5_1cat', 0))
    test_popularities['varied_5_1cat'] = max(generator_pop, test_popularities.get('varied_5_1cat', 0))

    # Test condition 9
    players = [generator] + same_cabs_4[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_4_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_4_1cat', False)
    test_ranks['same_4_1cat'] = max(generator_rank, test_ranks.get('same_4_1cat', 0))
    test_popularities['same_4_1cat'] = max(generator_pop, test_popularities.get('same_4_1cat', 0))

    # Test condition 10
    players = [generator] + same_cabs_5[1:-1] + [assassins[0]]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_5_1cat'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_5_1cat', False)
    test_ranks['same_5_1cat'] = max(generator_rank, test_ranks.get('same_5_1cat', 0))
    test_popularities['same_5_1cat'] = max(generator_pop, test_popularities.get('same_5_1cat', 0))

    # Test condition 11
    players = [generator] + varied_cabs_4[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_4_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_4_2cats', False)
    test_ranks['varied_4_2cats'] = max(generator_rank, test_ranks.get('varied_4_2cats', 0))
    test_popularities['varied_4_2cats'] = max(generator_pop, test_popularities.get('varied_4_2cats', 0))

    # Test condition 12
    players = [generator] + varied_cabs_5[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['varied_5_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('varied_5_2cats', False)
    test_ranks['varied_5_2cats'] = max(generator_rank, test_ranks.get('varied_5_2cats', 0))
    test_popularities['varied_5_2cats'] = max(generator_pop, test_popularities.get('varied_5_2cats', 0))

    # Test condition 13
    players = [generator] + same_cabs_4[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_4_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_4_2cats', False)
    test_ranks['same_4_2cats'] = max(generator_rank, test_ranks.get('same_4_2cats', 0))
    test_popularities['same_4_2cats'] = max(generator_pop, test_popularities.get('same_4_2cats', 0))

    # Test condition 14
    players = [generator] + same_cabs_5[1:-2] + assassins
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['same_5_2cats'] = True if final_popularities.max() == generator_pop else \
        test_results.get('same_5_2cats', False)
    test_ranks['same_5_2cats'] = max(generator_rank, test_ranks.get('same_5_2cats', 0))
    test_popularities['same_5_2cats'] = max(generator_pop, test_popularities.get('same_5_2cats', 0))

    # Test condition 15
    players = [generator] + trained_no_cat_gen0_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_no_cat_gen0'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_no_cat_gen0', False)
    test_ranks['trained_no_cat_gen0'] = max(generator_rank, test_ranks.get('trained_no_cat_gen0', 0))
    test_popularities['trained_no_cat_gen0'] = max(generator_pop, test_popularities.get('trained_no_cat_gen0', 0))

    # Test condition 16
    players = [generator] + trained_no_cat_gen49_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_no_cat_gen49'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_no_cat_gen49', False)
    test_ranks['trained_no_cat_gen49'] = max(generator_rank, test_ranks.get('trained_no_cat_gen49', 0))
    test_popularities['trained_no_cat_gen49'] = max(generator_pop, test_popularities.get('trained_no_cat_gen49', 0))

    # Test condition 17
    players = [generator] + trained_no_cat_gen99_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_no_cat_gen99'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_no_cat_gen99', False)
    test_ranks['trained_no_cat_gen99'] = max(generator_rank, test_ranks.get('trained_no_cat_gen99', 0))
    test_popularities['trained_no_cat_gen99'] = max(generator_pop, test_popularities.get('trained_no_cat_gen99', 0))

    # Test condition 18
    players = [generator] + trained_no_cat_gen149_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_no_cat_gen149'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_no_cat_gen149', False)
    test_ranks['trained_no_cat_gen149'] = max(generator_rank, test_ranks.get('trained_no_cat_gen149', 0))
    test_popularities['trained_no_cat_gen149'] = max(generator_pop, test_popularities.get('trained_no_cat_gen149', 0))

    # Test condition 19
    players = [generator] + trained_no_cat_gen199_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_no_cat_gen199'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_no_cat_gen199', False)
    test_ranks['trained_no_cat_gen199'] = max(generator_rank, test_ranks.get('trained_no_cat_gen199', 0))
    test_popularities['trained_no_cat_gen199'] = max(generator_pop, test_popularities.get('trained_no_cat_gen199', 0))

    # Test condition 20
    players = [generator] + trained_one_cat_gen0_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_one_cat_gen0'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_one_cat_gen0', False)
    test_ranks['trained_one_cat_gen0'] = max(generator_rank, test_ranks.get('trained_one_cat_gen0', 0))
    test_popularities['trained_one_cat_gen0'] = max(generator_pop, test_popularities.get('trained_one_cat_gen0', 0))

    # Test condition 21
    players = [generator] + trained_one_cat_gen49_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_one_cat_gen49'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_one_cat_gen49', False)
    test_ranks['trained_one_cat_gen49'] = max(generator_rank, test_ranks.get('trained_one_cat_gen49', 0))
    test_popularities['trained_one_cat_gen49'] = max(generator_pop, test_popularities.get('trained_one_cat_gen49', 0))

    # Test condition 22
    players = [generator] + trained_one_cat_gen99_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_one_cat_gen99'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_one_cat_gen99', False)
    test_ranks['trained_one_cat_gen99'] = max(generator_rank, test_ranks.get('trained_one_cat_gen99', 0))
    test_popularities['trained_one_cat_gen99'] = max(generator_pop, test_popularities.get('trained_one_cat_gen99', 0))

    # Test condition 23
    players = [generator] + trained_one_cat_gen149_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_one_cat_gen149'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_one_cat_gen149', False)
    test_ranks['trained_one_cat_gen149'] = max(generator_rank, test_ranks.get('trained_one_cat_gen149', 0))
    test_popularities['trained_one_cat_gen149'] = max(generator_pop, test_popularities.get('trained_one_cat_gen149', 0))

    # Test condition 24
    players = [generator] + trained_one_cat_gen199_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_one_cat_gen199'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_one_cat_gen199', False)
    test_ranks['trained_one_cat_gen199'] = max(generator_rank, test_ranks.get('trained_one_cat_gen199', 0))
    test_popularities['trained_one_cat_gen199'] = max(generator_pop, test_popularities.get('trained_one_cat_gen199', 0))

    # Test condition 25
    players = [generator] + trained_two_cat_gen0_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_two_cat_gen0'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_two_cat_gen0', False)
    test_ranks['trained_two_cat_gen0'] = max(generator_rank, test_ranks.get('trained_two_cat_gen0', 0))
    test_popularities['trained_two_cat_gen0'] = max(generator_pop, test_popularities.get('trained_two_cat_gen0', 0))

    # Test condition 26
    players = [generator] + trained_two_cat_gen49_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_two_cat_gen49'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_two_cat_gen49', False)
    test_ranks['trained_two_cat_gen49'] = max(generator_rank, test_ranks.get('trained_two_cat_gen49', 0))
    test_popularities['trained_two_cat_gen49'] = max(generator_pop, test_popularities.get('trained_two_cat_gen49', 0))

    # Test condition 27
    players = [generator] + trained_two_cat_gen99_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_two_cat_gen99'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_two_cat_gen99', False)
    test_ranks['trained_two_cat_gen99'] = max(generator_rank, test_ranks.get('trained_two_cat_gen99', 0))
    test_popularities['trained_two_cat_gen99'] = max(generator_pop, test_popularities.get('trained_two_cat_gen99', 0))

    # Test condition 28
    players = [generator] + trained_two_cat_gen149_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_two_cat_gen149'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_two_cat_gen149', False)
    test_ranks['trained_two_cat_gen149'] = max(generator_rank, test_ranks.get('trained_two_cat_gen149', 0))
    test_popularities['trained_two_cat_gen149'] = max(generator_pop, test_popularities.get('trained_two_cat_gen149', 0))

    # Test condition 29
    players = [generator] + trained_two_cat_gen199_cabs[:-1]
    assert len(players) == MAX_PLAYERS
    final_popularities = run_with_specified_agents(players)
    generator_pop = final_popularities[0]
    generator_rank = rankdata(final_popularities)[0]
    test_results['trained_two_cat_gen199'] = True if final_popularities.max() == generator_pop else \
        test_results.get('trained_two_cat_gen199', False)
    test_ranks['trained_two_cat_gen199'] = max(generator_rank, test_ranks.get('trained_two_cat_gen199', 0))
    test_popularities['trained_two_cat_gen199'] = max(generator_pop, test_popularities.get('trained_two_cat_gen199', 0))

print(test_results)
print(test_ranks)
print(test_popularities)
