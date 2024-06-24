import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from typing import List

GENES = [
    "alpha", "otherishDebtLimits", "coalitionTarget", "fixedUsage", "w_modularity", "w_centrality",
    "w_collective_strength", "w_familiarity", "w_prosocial", "initialDefense", "minKeep", "defenseUpdate",
    "defensePropensity", "fearDefense", "safetyFirst", "pillageFury", "pillageDelay", "pillagePriority",
    "pillageMargin", "pillageCompanionship", "pillageFriends", "vengenceMultiplier", "vengenceMax", "vengencePriority",
    "defendFriendMultiplier", "defendFriendMax", "defendFriendPriority", "attackGoodGuys", "limitingGive",
    "groupAware"
]
FOLDERS = [f'../ResultsSaved/no_cat/', f'../ResultsSaved/one_cat/', f'../ResultsSaved/two_cats/']


def find_optimal_kmeans(embeddings: np.array, show_plot: bool = False) -> KMeans:
    k_range, silhouette_scores, kmeans_models = range(2, 11), [], []

    for k in k_range:
        print(f'Calculating score for {k} clusters...')
        kmeans = KMeans(
            init="random",
            n_clusters=k,
            n_init='auto',
            random_state=1234
        )
        kmeans.fit(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
        kmeans_models.append(kmeans)
    print()

    if show_plot:
        plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
        plt.title('Silhouette Scores for Different Values of K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.xticks(k_range)
        plt.grid(True)
        plt.savefig(f'./plots/k_scores_trained.png', bbox_inches='tight')
        plt.clf()

    # Find the optimal k means model based on the highest silhouette score
    return kmeans_models[np.argmax(silhouette_scores)]


def show_embeddings(embeddings: np.array, kmeans: KMeans) -> None:
    labels = kmeans.predict(embeddings)
    color_template = plt.cm.tab10(np.linspace(0, 1, kmeans.n_clusters))
    colors = [color_template[x] for x in list(labels)]
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=100, alpha=0.8, color=colors)
    plt.title('CAB Clusters')
    plt.savefig(f'./plots/cab_clusters_trained.png', bbox_inches='tight')
    plt.clf()


def cluster_averages(original_samples: List, scores: List, kmeans: KMeans) -> None:
    labels, cluster_data, cluster_scores = kmeans.labels_, {}, {}
    assert len(original_samples) == len(labels)

    for i in range(len(original_samples)):
        cluster = labels[i]
        cluster_data[cluster] = cluster_data.get(cluster, []) + [original_samples[i]]
        cluster_scores[cluster] = cluster_scores.get(cluster, []) + [scores[i]]

    for cluster, cluster_samples in cluster_data.items():
        scores = cluster_scores[cluster]
        weighted_samples = np.dot(scores, cluster_samples)
        total_weight = np.sum(scores)
        cluster_sample_avg = weighted_samples / total_weight
        print(f'Cluster {cluster}: ')
        for i, gene in enumerate(GENES):
            gene_avg = round(cluster_sample_avg[i])
            print(f'{gene} = {gene_avg}')
        print()


print('Generating samples...')
samples, scores = [], []

for folder in FOLDERS:
    files = os.listdir(folder)

    for file in files:
        df = pd.read_csv(f'{folder}{file}', header=None)

        for i in range(len(df)):
            gene, score = df.iloc[i, 0], df.iloc[i, -1]
            gene_vals = gene.split('_')[3:-1]
            assert len(gene_vals) == len(GENES)
            samples.append([int(val) for val in gene_vals])
            scores.append(float(score))

print('Reducing to two dimensions...')
# embeddings = TSNE(n_components=2).fit_transform(np.array(samples))
embeddings = samples
# kmeans = find_optimal_kmeans(embeddings, show_plot=True)

kmeans = KMeans(
    init="random",
    n_clusters=16,
    n_init='auto',
    random_state=1234
)
kmeans.fit(embeddings)

# show_embeddings(embeddings, kmeans)
cluster_averages(samples, scores, kmeans)
