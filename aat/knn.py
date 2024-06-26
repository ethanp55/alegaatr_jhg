import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# Read in the training data
generator_to_alignment_vectors, generator_to_correction_terms = {}, {}
training_data_folder = '../aat/training_data/'

for file in os.listdir(training_data_folder):
    generator_idx = file.split('_')[1]
    data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)
    is_alignment_vectors = 'vectors' in file
    map_to_add_to = generator_to_alignment_vectors if is_alignment_vectors else generator_to_correction_terms
    map_to_add_to[generator_idx] = data

# Make sure the training data was read in properly
for generator_idx, vectors in generator_to_alignment_vectors.items():
    correction_terms = generator_to_correction_terms[generator_idx]

    assert len(vectors) == len(correction_terms)

# Train KNN models for each generator
for generator_idx, x in generator_to_alignment_vectors.items():
    y = generator_to_correction_terms[generator_idx]
    # pca = PCA(n_components=0.95)
    # x_reduced = pca.fit_transform(x)
    x_reduced = x

    print(f'X and Y data for generator {generator_idx}')
    print('X train shape: ' + str(x_reduced.shape))
    print('Y train shape: ' + str(y.shape))

    # Use cross validation (10 folds) to find the best k value
    k_values, cv_scores = range(1, int(len(x_reduced) ** 0.5) + 1), []
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, x_reduced, y, cv=10)
        cv_scores.append(scores.mean())
    n_neighbors = k_values[np.argmax(cv_scores)]
    print(f'Best R-squared: {cv_scores[np.argmax(cv_scores)]}')
    print(f'N neighbors: {n_neighbors}\n')

    # Create and store the model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(x_reduced, y)

    with open(f'../aat/knn_models/generator_{generator_idx}_knn.pickle', 'wb') as f:
        pickle.dump(knn, f)

    # with open(f'../aat/knn_models/generator_{generator_idx}_pca.pickle', 'wb') as f:
    #     pickle.dump(pca, f)
