import numpy as np
import os
from scipy.stats import hmean

baselines = {'keep_all': 81.79069375972314, 'equal_steal': 223.8113639516392, 'coop': 247.25596023343223}
results, folder = {}, '../simulations/adaptability_results/'

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if agent_name not in results:
        results[agent_name] = {}
    opp_type = file[len(agent_name) + 1:-4]
    comparison = baselines[opp_type]
    data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)

    if opp_type == 'keep_all' or opp_type == 'equal_steal':
        avg_final_pop = sum([row[-1] for row in data]) / len(data)

    elif opp_type == 'coop':
        row_avgs = [sum(row) / len(row) for row in data]
        avg_final_pop = sum(row_avgs) / len(row_avgs)

    else:
        raise Exception(f'{opp_type} is not a defined opponent type')

    regret = max(comparison - avg_final_pop, 0)
    val = 1 - (regret / comparison)

    assert 0 <= val <= 1
    results[agent_name][opp_type] = val

rc_scores = []
for agent, res, in results.items():
    print(agent)
    defect_scores, coop_score = [], -1
    for opp_type in baselines.keys():
        print(f'{opp_type}: {res[opp_type]}')
        if opp_type != 'coop':
            defect_scores.append(res[opp_type])
        else:
            coop_score = res[opp_type]
    defect_score = hmean(defect_scores)
    robust_coop_score = min(defect_score, coop_score)
    print(f'Defect score: {defect_score}')
    print(f'Coop score: {coop_score}')
    print(f'Robust coop score: {robust_coop_score}\n')
    rc_scores.append((agent, robust_coop_score))
rc_scores.sort(key=lambda x: x[1], reverse=True)
print(rc_scores)
