from GeneSimulation_py.geneagent3 import GeneAgent3
from GeneSimulation_py.main import run_with_specified_agents
import pandas as pd

generators, generator_df = [], pd.read_csv(f'../ResultsSaved/generator_genes/genes.csv', header=None)

for i in range(len(generator_df)):
    gene_str = generator_df.iloc[i, 0]
    generators.append(GeneAgent3(gene_str, 1, check_assumptions=True))

generators.append(GeneAgent3(gene_str, 1, check_assumptions=True))
players = generators[len(generators) - 10:]
assert len(players) == 10
run_with_specified_agents(players, numRounds=100)
