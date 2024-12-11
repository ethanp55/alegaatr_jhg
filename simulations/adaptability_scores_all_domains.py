import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hmean

USE_H_MEAN = False

# ----------------------------------------------------------------------------------------------------------------------
# SCORES BY LEARNING ALGORITHM
# ----------------------------------------------------------------------------------------------------------------------
regaetune_d_scores = [0.6765503819402352, 1.0, 0.6998287617026768,
                      0.5, 0.8333333333333334, 0.6,
                      0.987069390978186, 0.39482775639127443, 0.39482775639127443]
regaetune_sp_scores = [0.4442681410717302, 0.3307936184126976, 0.7588455817346218,
                       0.4, 0.8333333333333334, 1.0,
                       0.0, 0.1563006596793094, 0.8518262897987787]
regaetune_c_scores = [0.558449805042446, 0.11380900408827321, 0.5799642776305467,
                      0.9388888888888889, 0.7555555555555555, 0.3722222222222221,
                      0.34890775725085477, 0.5254642633581048, 0.4773124889652185]
regaetune_adapt_scores = [0.4442681410717302, 0.11380900408827321, 0.5799642776305467,
                          0.4, 0.7555555555555555, 0.3722222222222221,
                          0.0, 0.1563006596793094, 0.39482775639127443]

alegaatr_d_scores = [0.8460211223125407,
                     0.9333333333333333,
                     0.6415951041358208]
alegaatr_sp_scores = [0.5663651352423535,
                      0.8333333333333334,
                      0.744822346703476]
alegaatr_c_scores = [0.5198144176254589,
                     0.8055555555555556,
                     0.4131101231080365]
alegaatr_adapt_scores = [0.5198144176254589,
                         0.8055555555555556,
                         0.4131101231080365]

traditional_d_scores = [0.7789888408054728, 1.0, 0.560334796856711,
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 0.6909485736847303]
traditional_sp_scores = [0.3307936184126976, 0.3307936184126976, 0.684377727369361,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0]
traditional_c_scores = [0.0034082829107250356, 0.3079225535431822, 0.6454426824200833,
                        0.4375, 0.4375, 0.4375,
                        0.26865479992937735, 0.34890775725085477, 0.3970595316437411]
traditional_adapt_scores = [0.0034082829107250356, 0.3079225535431822, 0.21535266842312414,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]

r_d_avg = hmean(regaetune_d_scores) if USE_H_MEAN else sum(regaetune_d_scores) / len(regaetune_d_scores)
r_sp_avg = hmean(regaetune_sp_scores) if USE_H_MEAN else sum(regaetune_sp_scores) / len(regaetune_sp_scores)
r_c_avg = hmean(regaetune_c_scores) if USE_H_MEAN else sum(regaetune_c_scores) / len(regaetune_c_scores)
r_a_avg = hmean(regaetune_adapt_scores) if USE_H_MEAN else sum(regaetune_adapt_scores) / len(regaetune_adapt_scores)

aleg_d_avg = hmean(alegaatr_d_scores) if USE_H_MEAN else sum(alegaatr_d_scores) / len(alegaatr_d_scores)
aleg_sp_avg = hmean(alegaatr_sp_scores) if USE_H_MEAN else sum(alegaatr_sp_scores) / len(alegaatr_sp_scores)
aleg_c_avg = hmean(alegaatr_c_scores) if USE_H_MEAN else sum(alegaatr_c_scores) / len(alegaatr_c_scores)
aleg_a_avg = hmean(alegaatr_adapt_scores) if USE_H_MEAN else sum(alegaatr_adapt_scores) / len(alegaatr_adapt_scores)

t_d_avg = hmean(traditional_d_scores) if USE_H_MEAN else sum(traditional_d_scores) / len(traditional_d_scores)
t_sp_avg = hmean(traditional_sp_scores) if USE_H_MEAN else sum(traditional_sp_scores) / len(traditional_sp_scores)
t_c_avg = hmean(traditional_c_scores) if USE_H_MEAN else sum(traditional_c_scores) / len(traditional_c_scores)
t_a_avg = hmean(traditional_adapt_scores) if USE_H_MEAN else sum(traditional_adapt_scores) / len(
    traditional_adapt_scores)

print('REGAETUNE:')
print(f'Defect = {r_d_avg}')
print(f'Self-play = {r_sp_avg}')
print(f'Coop = {r_c_avg}')
print(f'Adaptability = {r_a_avg}\n')

print('ALEGAATR:')
print(f'Defect = {aleg_d_avg}')
print(f'Self-play = {aleg_sp_avg}')
print(f'Coop = {aleg_c_avg}')
print(f'Adaptability = {aleg_a_avg}\n')

print('TRADITIONAL:')
print(f'Defect = {t_d_avg}')
print(f'Self-play = {t_sp_avg}')
print(f'Coop = {t_c_avg}')
print(f'Adaptability = {t_a_avg}\n')

algorithms = ['Traditional', 'REGaeTune', 'AlegAATr']
conditions = ['Defect', 'Self-Play', 'Cooperate', 'Adaptability']
df = pd.DataFrame(
    {
        algorithms[0]: [traditional_d_scores, traditional_sp_scores, traditional_c_scores, traditional_adapt_scores],
        algorithms[1]: [regaetune_d_scores, regaetune_sp_scores, regaetune_c_scores, regaetune_adapt_scores],
        algorithms[2]: [alegaatr_d_scores, alegaatr_sp_scores, alegaatr_c_scores, alegaatr_adapt_scores]
    }
)


def calculate_stats(results_list):
    means = [np.mean(results) for results in results_list]
    std_errors = [np.std(results, ddof=1) / np.sqrt(len(results)) for results in results_list]
    return means, std_errors


stats = {alg: calculate_stats(df[alg]) for alg in algorithms}
for alg in algorithms:
    df[f'{alg}_mean'], df[f'{alg}_se'] = stats[alg]
mean_values = df[[f'{alg}_mean' for alg in algorithms]].values
se_values = df[[f'{alg}_se' for alg in algorithms]].values
x = np.arange(len(conditions))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 3))
plt.grid()
bars1 = ax.bar(x - width, mean_values[:, 0], width, yerr=se_values[:, 0], label=algorithms[0], capsize=5,
               color='crimson')
bars2 = ax.bar(x, mean_values[:, 1], width, yerr=se_values[:, 1], label=algorithms[1], capsize=5, color='gold')
bars3 = ax.bar(x + width, mean_values[:, 2], width, yerr=se_values[:, 2], label=algorithms[2], capsize=5,
               color='lightblue')
ax.set_xlabel('Condition', fontsize=18, fontweight='bold')
ax.set_ylabel('Score', fontsize=18, fontweight='bold')
# ax.set_title('Algorithm Performance by Condition')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend()
plt.savefig('../simulations/scores_by_learning_alg.png', bbox_inches='tight')
plt.clf()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# SCORES BY FEATURES
# ----------------------------------------------------------------------------------------------------------------------

raw_d_scores = [0.7789888408054728, 0.6765503819402352,
                1.0, 0.5,
                1.0, 0.987069390978186]
raw_sp_scores = [0.3307936184126976, 0.4442681410717302,
                 0.0, 0.4,
                 0.0, 0.0]
raw_c_scores = [0.0034082829107250356, 0.558449805042446,
                0.4375, 0.9388888888888889,
                0.26865479992937735, 0.34890775725085477]
raw_a_scores = [0.0034082829107250356, 0.4442681410717302,
                0.0, 0.4,
                0.0, 0.0]

aat_d_scores = [1.0, 1.0,
                1.0, 0.8333333333333334,
                1.0, 0.39482775639127443]
aat_sp_scores = [0.3307936184126976, 0.3307936184126976,
                 0.0, 0.8333333333333334,
                 0.0, 0.1563006596793094]
aat_c_scores = [0.3079225535431822, 0.11380900408827321,
                0.4375, 0.7555555555555555,
                0.34890775725085477, 0.5254642633581048]
aat_a_scores = [0.3079225535431822, 0.11380900408827321,
                0.0, 0.7555555555555555,
                0.0, 0.1563006596793094]

aata_d_scores = [1.0, 1.0, 0.8460211223125407,
                 1.0, 0.8333333333333334, 0.9333333333333333,
                 1.0, 0.39482775639127443, 0.6415951041358208]
aata_sp_scores = [0.3307936184126976, 0.3307936184126976, 0.5663651352423535,
                  0.0, 0.8333333333333334, 0.8333333333333334,
                  0.0, 0.1563006596793094, 0.744822346703476]
aata_c_scores = [0.3079225535431822, 0.11380900408827321, 0.5198144176254589,
                 0.4375, 0.7555555555555555, 0.8055555555555556,
                 0.34890775725085477, 0.5254642633581048, 0.4131101231080365]
aata_a_scores = [0.3079225535431822, 0.11380900408827321, 0.5198144176254589,
                 0.0, 0.7555555555555555, 0.8055555555555556,
                 0.0, 0.1563006596793094, 0.4131101231080365]

rawaat_d_scores = [0.560334796856711, 0.6998287617026768,
                   1.0, 0.6,
                   0.6909485736847303, 0.39482775639127443]
rawaat_sp_scores = [0.684377727369361, 0.7588455817346218,
                    0.0, 1.0,
                    0.0, 0.8518262897987787]
rawaat_c_scores = [0.6454426824200833, 0.5799642776305467,
                   0.4375, 0.3722222222222221,
                   0.3970595316437411, 0.4773124889652185]
rawaat_a_scores = [0.560334796856711, 0.5799642776305467,
                   0.0, 0.3722222222222221,
                   0.0, 0.39482775639127443]

raw_d_avg = hmean(raw_d_scores) if USE_H_MEAN else sum(raw_d_scores) / len(raw_d_scores)
raw_sp_avg = hmean(raw_sp_scores) if USE_H_MEAN else sum(raw_sp_scores) / len(raw_sp_scores)
raw_c_avg = hmean(raw_c_scores) if USE_H_MEAN else sum(raw_c_scores) / len(raw_c_scores)
raw_a_avg = hmean(raw_a_scores) if USE_H_MEAN else sum(raw_a_scores) / len(raw_a_scores)

aat_d_avg = hmean(aat_d_scores) if USE_H_MEAN else sum(aat_d_scores) / len(aat_d_scores)
aat_sp_avg = hmean(aat_sp_scores) if USE_H_MEAN else sum(aat_sp_scores) / len(aat_sp_scores)
aat_c_avg = hmean(aat_c_scores) if USE_H_MEAN else sum(aat_c_scores) / len(aat_c_scores)
aat_a_avg = hmean(aat_a_scores) if USE_H_MEAN else sum(aat_a_scores) / len(aat_a_scores)

aata_d_avg = hmean(aata_d_scores) if USE_H_MEAN else sum(aata_d_scores) / len(aata_d_scores)
aata_sp_avg = hmean(aata_sp_scores) if USE_H_MEAN else sum(aata_sp_scores) / len(aata_sp_scores)
aata_c_avg = hmean(aata_c_scores) if USE_H_MEAN else sum(aata_c_scores) / len(aata_c_scores)
aata_a_avg = hmean(aata_a_scores) if USE_H_MEAN else sum(aata_a_scores) / len(aata_a_scores)

rawaat_d_avg = hmean(rawaat_d_scores) if USE_H_MEAN else sum(rawaat_d_scores) / len(rawaat_d_scores)
rawaat_sp_avg = hmean(rawaat_sp_scores) if USE_H_MEAN else sum(rawaat_sp_scores) / len(rawaat_sp_scores)
rawaat_c_avg = hmean(rawaat_c_scores) if USE_H_MEAN else sum(rawaat_c_scores) / len(rawaat_c_scores)
rawaat_a_avg = hmean(rawaat_a_scores) if USE_H_MEAN else sum(rawaat_a_scores) / len(rawaat_a_scores)

print('RAW:')
print(f'Defect = {raw_d_avg}')
print(f'Self-play = {raw_sp_avg}')
print(f'Coop = {raw_c_avg}')
print(f'Adaptability = {raw_a_avg}\n')

print('AAT:')
print(f'Defect = {aat_d_avg}')
print(f'Self-play = {aat_sp_avg}')
print(f'Coop = {aat_c_avg}')
print(f'Adaptability = {aat_a_avg}')
print('---------------------------')
print(f'Defect = {aata_d_avg}')
print(f'Self-play = {aata_sp_avg}')
print(f'Coop = {aata_c_avg}')
print(f'Adaptability = {aata_a_avg}\n')

print('RAWAAT:')
print(f'Defect = {rawaat_d_avg}')
print(f'Self-play = {rawaat_sp_avg}')
print(f'Coop = {rawaat_c_avg}')
print(f'Adaptability = {rawaat_a_avg}')

features = ['Raw', 'AAT', 'RawAAT']
conditions = ['Defect', 'Self-Play', 'Cooperate', 'Adaptability']
df = pd.DataFrame(
    {
        features[0]: [raw_d_scores, raw_sp_scores, raw_c_scores, raw_a_scores],
        features[1]: [aat_d_scores, aat_sp_scores, aat_c_scores, aat_a_scores],
        features[2]: [rawaat_d_scores, rawaat_sp_scores, rawaat_c_scores, rawaat_a_scores]
    }
)
stats = {feat: calculate_stats(df[feat]) for feat in features}
for feat in features:
    df[f'{feat}_mean'], df[f'{feat}_se'] = stats[feat]
mean_values = df[[f'{feat}_mean' for feat in features]].values
se_values = df[[f'{feat}_se' for feat in features]].values
x = np.arange(len(conditions))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 3))
plt.grid()
bars1 = ax.bar(x - width, mean_values[:, 0], width, yerr=se_values[:, 0], label=features[0], capsize=5, color='purple')
bars2 = ax.bar(x, mean_values[:, 1], width, yerr=se_values[:, 1], label=features[1], capsize=5, color='teal')
bars3 = ax.bar(x + width, mean_values[:, 2], width, yerr=se_values[:, 2], label=features[2], capsize=5, color='gray')
ax.set_xlabel('Condition', fontsize=18, fontweight='bold')
ax.set_ylabel('Score', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend()
plt.savefig('../simulations/scores_by_feature_set.png', bbox_inches='tight')
plt.clf()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
