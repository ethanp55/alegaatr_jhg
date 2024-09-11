import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from typing import List, Optional


def _plot_embeddings(labels: List[str], embeddings: np.array, agent_name: str, n_cats: Optional[int] = None,
                     color_by_generator: bool = False) -> None:
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('tab20')(Normalize()(unique_labels))
    plt.figure(figsize=(15, 15))

    if color_by_generator:
        for j, label in enumerate(unique_labels):
            label_points = embeddings[labels == label]
            plt.scatter(label_points[:, 0], label_points[:, 1], s=50, alpha=1.0, color=colors[j], label=label)

        plt.legend(loc='best', fontsize='18')

    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], s=50, alpha=1.0, color='green')

    image_name_adj = f'_c={n_cats}' if n_cats is not None else ''
    plt.savefig(f'../simulations/vector_plots/{agent_name}{image_name_adj}.png', bbox_inches='tight')
    plt.clf()


def _evaluate_clustering(embeddings: np.array, agent_name: str) -> None:
    k_range, silhouette_scores, db_scores, wcss_scores, bcss_scores = range(20, 41), [], [], [], []
    overall_mean = np.mean(embeddings, axis=0)

    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        silhouette_scores.append(silhouette_score(embeddings, labels))

        db_scores.append(davies_bouldin_score(embeddings, labels))

        wcss = 0
        for label in np.unique(labels):
            cluster_points = embeddings[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            wcss += np.sum((cluster_points - cluster_center) ** 2)
        wcss_scores.append(wcss)

        bcss = 0
        for label in np.unique(labels):
            cluster_points = embeddings[labels == label]
            cluster_mean = np.mean(cluster_points, axis=0)
            n_points = cluster_points.shape[0]
            distance = np.sum((cluster_mean - overall_mean) ** 2)
            bcss += n_points * distance
        bcss_scores.append(bcss)

    best_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_idx]
    best_sil = silhouette_scores[best_idx]
    best_db = db_scores[best_idx]
    best_wcss = wcss_scores[best_idx]
    best_bcss = bcss_scores[best_idx]

    print(f'{agent_name}, num clusters = {best_k}')
    print(f'silhouette score = {best_sil} (higher is better)')
    print(f'DB score = {best_db} (lower is better)')
    print(f'WCSS = {best_wcss} (lower is better)')
    print(f'BCSS = {best_bcss} (higher is better)\n')


folder = '../simulations/vectors/'
alegaatr_0_cats, alegaatr_1_cat, alegaatr_2_cats = {}, {}, {}
dqn_0_cats, dqn_1_cat, dqn_2_cats = {}, {}, {}

for file in os.listdir(folder):
    file_path = f'{folder}{file}'
    data = np.genfromtxt(file_path, delimiter=',', skip_header=0)
    generators, vectors = data[:, 0], data[:, 1:]
    assert generators.shape[0] == vectors.shape[0]

    if 'AlegAATr' in file:
        dict_to_use = alegaatr_0_cats if 'c=0' in file else (alegaatr_1_cat if 'c=1' in file else alegaatr_2_cats)

    else:
        dict_to_use = dqn_0_cats if 'c=0' in file else (dqn_1_cat if 'c=1' in file else dqn_2_cats)

    for i in range(generators.shape[0]):
        generator_idx = int(generators[i])
        dict_to_use[generator_idx] = dict_to_use.get(generator_idx, []) + [vectors[i, :]]

for agent_name in ['AlegAATr', 'DQN']:
    all_labels, all_vectors = [], None
    dict_list = [alegaatr_0_cats, alegaatr_1_cat, alegaatr_2_cats] if agent_name == 'AlegAATr' else \
        [dqn_0_cats, dqn_1_cat, dqn_2_cats]

    for i, dict in enumerate(dict_list):
        labels, vectors = [], None

        for key, vector_list in dict.items():
            labels.extend([key] * len(vector_list))
            all_labels.extend([key] * len(vector_list))
            vectors = np.array(vector_list) if vectors is None else np.concatenate([vectors, np.array(vector_list)])
            all_vectors = np.array(vector_list) if all_vectors is None else np.concatenate(
                [all_vectors, np.array(vector_list)])

        _plot_embeddings(labels, TSNE(n_components=2).fit_transform(vectors), agent_name, i, color_by_generator=False)

    all_embeddings = TSNE(n_components=2).fit_transform(all_vectors)
    _plot_embeddings(all_labels, all_embeddings, agent_name, color_by_generator=False)
    _evaluate_clustering(all_embeddings, agent_name)
