import os
import pandas as pd

FOLDER = '../ResultsSaved/no_cat/'
DIFF_THRESHOLD = 60

file_list = os.listdir(FOLDER)
file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by generation number

results = []

for i in range(1, len(file_list)):
    curr_file, prev_file = file_list[i], file_list[i - 1]
    curr_df = pd.read_csv(f'{FOLDER}{curr_file}', header=None)
    prev_df = pd.read_csv(f'{FOLDER}{prev_file}', header=None)
    curr_pop, prev_pop = curr_df.iloc[0, -1], prev_df.iloc[0, -1]
    diff = curr_pop - prev_pop

    if diff >= DIFF_THRESHOLD:
        results.append((curr_file, curr_pop, prev_pop, diff))

results.sort(key=lambda x: x[-1], reverse=True)

print(f'Total: {len(results)}')

for result in results:
    print(result)
