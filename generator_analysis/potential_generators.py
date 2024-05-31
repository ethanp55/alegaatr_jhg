import os
import pandas as pd

FOLDER = '../ResultsSaved/no_cat/'
DIFF_THRESHOLD = 60

file_list = os.listdir(FOLDER)
results = []

for file in file_list:
    df = pd.read_csv(f'{FOLDER}{file}', header=None)
    best = df.iloc[0, -1]
    second_best = df.iloc[1, -1]
    diff = best - second_best

    if diff >= DIFF_THRESHOLD:
        results.append((file, best, second_best, diff))

results.sort(key=lambda x: x[-1], reverse=True)

print(f'Total: {len(results)}')

for result in results:
    print(result)
