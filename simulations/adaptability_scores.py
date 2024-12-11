import numpy as np
import os
from scipy.stats import hmean

baselines = {'keep_all': 81.79069375972314, 'equal_steal': 223.8113639516392, 'selfplay': 247.25596023343223,
             'coop': 247.25596023343223}
results, folder = {}, '../simulations/adaptability_results/'
minimax_val, lowest_reward = 0, 0

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if agent_name not in results:
        results[agent_name] = {}
    opp_type = file[len(agent_name) + 1:-4]
    comparison = baselines[opp_type]
    data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)

    if opp_type == 'keep_all' or opp_type == 'equal_steal':
        avg_pop = sum([row[-1] for row in data]) / len(data)
        regret = (comparison - lowest_reward) - (avg_pop - lowest_reward)
        val = 1 - max((regret / (comparison - lowest_reward)), 0)

    elif opp_type == 'selfplay':
        row_avgs = [sum(row) / len(row) for row in data]
        avg_pop = sum(row_avgs) / len(row_avgs)
        regret = (comparison - minimax_val) - (avg_pop - minimax_val)
        val = 1 - min((regret / (comparison - minimax_val)), 1)

    elif opp_type == 'coop':
        avg_pop = sum([row[-1] for row in data]) / len(data)
        regret = (comparison - minimax_val) - (avg_pop - minimax_val)
        val = 1 - min((regret / (comparison - minimax_val)), 1)

    else:
        raise Exception(f'{opp_type} is not a defined opponent type')

    assert 0 <= val <= 1
    results[agent_name][opp_type] = val

rc_scores = []
for agent, res, in results.items():
    print(agent)
    defect_scores, self_play_score, coop_score = [], -1, -1
    for opp_type in baselines.keys():
        print(f'{opp_type}: {res[opp_type]}')
        if opp_type == 'selfplay':
            self_play_score = res[opp_type]
        elif opp_type == 'coop':
            coop_score = res[opp_type]
        else:
            defect_scores.append(res[opp_type])
    # defect_score = hmean(defect_scores)
    defect_score = sum(defect_scores) / len(defect_scores)
    robust_coop_score = min([defect_score, self_play_score, coop_score])
    print(f'Defect score: {defect_score}')
    print(f'Self-play score: {self_play_score}')
    print(f'Coop score: {coop_score}')
    print(f'Robust coop score: {robust_coop_score}\n')
    rc_scores.append((agent, robust_coop_score))
rc_scores.sort(key=lambda x: x[1], reverse=True)
print(rc_scores)
