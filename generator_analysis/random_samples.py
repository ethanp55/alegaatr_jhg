import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple

N_SAMPLES = 100000
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
    k_range, silhouette_scores, kmeans_models = range(2, 26), [], []

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
        plt.savefig(f'./plots/k_scores.png', bbox_inches='tight')
        plt.clf()

    # Find the optimal k means model based on the highest silhouette score
    return kmeans_models[np.argmax(silhouette_scores)]


def show_embeddings(embeddings: np.array, kmeans: KMeans, training_samples: List[List[int]],
                    training_colors: Dict[int, str]) -> None:
    labels = kmeans.predict(embeddings)
    color_template = plt.cm.tab10(np.linspace(0, 1, kmeans.n_clusters))
    colors = [color_template[x] for x in list(labels[:len(embeddings) - len(training_samples)])]
    training_sample_colors = [training_colors[i] for i in range(len(training_samples))]
    colors += training_sample_colors
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=100, alpha=0.8, color=colors)
    plt.title('CAB Clusters')
    plt.savefig(f'./plots/cab_clusters.png', bbox_inches='tight')
    plt.clf()


def cluster_averages(original_samples: List, kmeans: KMeans) -> None:
    labels, cluster_data = kmeans.labels_, {}
    assert len(original_samples) == len(labels)

    for i in range(len(original_samples)):
        cluster = labels[i]
        cluster_data[cluster] = cluster_data.get(cluster, []) + [original_samples[i]]

    for cluster, cluster_samples in cluster_data.items():
        cluster_sample_avg = np.array(cluster_samples).mean(axis=0)
        print(f'Cluster {cluster}: ')
        for i, gene in enumerate(GENES):
            gene_avg = round(cluster_sample_avg[i])
            print(f'{gene} = {gene_avg}')
        print()


def sample_from_clusters(original_samples: List, kmeans: KMeans, samples_per_cluster: int = 1,
                         save: bool = False) -> None:
    labels, cluster_data, gene_strings, vectors = kmeans.labels_, {}, [], []
    assert len(original_samples) == len(labels)

    for i in range(len(original_samples)):
        cluster = labels[i]
        cluster_data[cluster] = cluster_data.get(cluster, []) + [original_samples[i]]

    for cluster, cluster_samples in cluster_data.items():
        sampled_indices = np.random.choice(range(len(cluster_samples)), samples_per_cluster, replace=False)
        print(f'Cluster {cluster}: ')
        for idx in sampled_indices:
            vec = cluster_samples[idx]
            vectors.append(vec)
            for i, gene in enumerate(GENES):
                gene_val = round(vec[i])
                print(f'{gene} = {gene_val}')

            if save:
                vec_str = list(vec)
                vec_str = [str(val) for val in vec_str]
                gene_str = 'gene_' + '_'.join(vec_str) + ',0,0,0'
                gene_strings.append(gene_str)
            print()

    print(np.array(vectors).mean(axis=0))

    if save:
        csv_data = str.join('\n', gene_strings)
        with open('../ResultsSaved/generator_genes/genes.csv', 'w') as f:
            f.write(csv_data)


def samples_from_training() -> Tuple[List[List[int]], Dict[int, str]]:
    training_samples, training_colors, i = [], {}, 0

    for folder in FOLDERS:
        files = os.listdir(folder)

        for file in files:
            df = pd.read_csv(f'{folder}{file}', header=None)
            gene, score = df.iloc[0, 0], df.iloc[0, -1]
            gene_vals = gene.split('_')[3:-1]
            assert len(gene_vals) == len(GENES)
            training_samples.append([int(val) for val in gene_vals])

            if 'no_cat' in folder:
                training_colors[i] = 'black'

            elif 'one_cat' in folder:
                training_colors[i] = 'cyan'

            else:
                training_colors[i] = 'purple'

            i += 1

    return training_samples, training_colors


print('Generating samples...')
samples, possible_vals = [], range(0, 101)

for _ in range(N_SAMPLES):
    sample = [np.random.choice(possible_vals) for _ in GENES]
    samples.append(sample)

# training_samples, training_colors = samples_from_training()
training_samples, training_colors = [], {}

print('Reducing to two dimensions...')
# embeddings = TSNE(n_components=2).fit_transform(np.array(samples + training_samples))
embeddings = np.array(samples + training_samples)
# kmeans = find_optimal_kmeans(embeddings, show_plot=True)

kmeans = KMeans(
    init="random",
    n_clusters=16,
    n_init='auto',
    random_state=1234
)
kmeans.fit(embeddings)

# show_embeddings(embeddings, kmeans, training_samples, training_colors)
# cluster_averages(samples, kmeans)
sample_from_clusters(samples, kmeans, save=True)
