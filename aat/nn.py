import numpy as np
import os
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Read in the training data
generator_to_alignment_vectors, generator_to_correction_terms = {}, {}
training_data_folder = '../aat/training_data/'
enhanced = True
adjustment = '_enh' if enhanced else ''

for file in os.listdir(training_data_folder):
    if (enhanced and '_enh' not in file) or (not enhanced and '_enh' in file):
        continue

    generator_idx = file.split('_')[1]
    data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)
    is_alignment_vectors = 'vectors' in file
    map_to_add_to = generator_to_alignment_vectors if is_alignment_vectors else generator_to_correction_terms
    map_to_add_to[generator_idx] = data

# Make sure the training data was read in properly
for generator_idx, vectors in generator_to_alignment_vectors.items():
    correction_terms = generator_to_correction_terms[generator_idx]

    assert len(vectors) == len(correction_terms)

# Train neural networks for each generator
for generator_idx, x in generator_to_alignment_vectors.items():
    # Get training and validation data
    y = generator_to_correction_terms[generator_idx]

    # Scale the input data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Try different hyperparameters and pick the best set (cross validation with 5 folds)
    print(f'Training generator {generator_idx}')
    print('X train shape: ' + str(x_scaled.shape))
    print('Y train shape: ' + str(y.shape))
    param_grid = {
        'hidden_layer_sizes': [(25, 50), (50, 100), (25, 25), (50, 50), (100, 100), (25, 50, 25), (100, 150, 100)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
    }
    random_search = RandomizedSearchCV(MLPRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', n_iter=5)
    random_search.fit(x_scaled, y)
    mlp = random_search.best_estimator_
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best MSE: {-random_search.best_score_}')
    print(f'Best R-squared: {r2_score(y, mlp.predict(x_scaled))}\n')

    # Store the best model and the scaler
    with open(f'../aat/nn_models/generator_{generator_idx}_nn{adjustment}.pickle', 'wb') as f:
        pickle.dump(mlp, f)

    with open(f'../aat/nn_models/generator_{generator_idx}_scaler{adjustment}.pickle', 'wb') as f:
        pickle.dump(scaler, f)
