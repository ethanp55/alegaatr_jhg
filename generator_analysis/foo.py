import pandas as pd

df = pd.read_csv('../aat/training_data/generator_0_vectors.csv', header=None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)
print(df.describe())
