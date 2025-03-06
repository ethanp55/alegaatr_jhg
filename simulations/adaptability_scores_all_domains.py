import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

jhg_results = pd.read_csv('./jhg_adaptability_results.csv')
prisoners_results = pd.read_csv('./prisoners_adaptability_results.csv')
pursuit_results = pd.read_csv('./pursuit_adaptability_results.csv')

df = pd.concat([jhg_results, prisoners_results, pursuit_results], ignore_index=True)

# The interaction effect is not significant, so we're just using an additive model
include_alegaatr = False
score_type = 'a'
df_filtered = df[(df['score_type'] == score_type) & (df['learning_alg'] != 'REGAEKNN') & (
        df['feature_set'] != 'AATKNN')] if not include_alegaatr else df[df['score_type'] == score_type]
model = ols('score ~ C(learning_alg) + C(feature_set)', data=df_filtered).fit()
anova = sm.stats.anova_lm(model, typ=2)
print(anova)
print()

# Pairwise comparisons - features
print(pairwise_tukeyhsd(endog=df_filtered['score'], groups=df_filtered['feature_set'], alpha=0.05))
print()

# Pairwise comparisons - learning alg
print(pairwise_tukeyhsd(endog=df_filtered['score'], groups=df_filtered['learning_alg'], alpha=0.05))
print()

# Score averages and standard errors across the three domains
for domain in ['jhg', 'prisoners', 'pursuit']:
    print(domain)
    result = df[df['domain'] == domain].groupby(['domain', 'score_type', 'learning_alg', 'feature_set']).agg(
        average_score=('score', 'mean'),
        std_dev=('score', 'std'),
        count=('score', 'count')
    ).reset_index()
    result['standard_error'] = result['std_dev'] / np.sqrt(result['count'])
    print(result[['score_type', 'learning_alg', 'feature_set', 'average_score', 'standard_error', 'count']])
    print()
