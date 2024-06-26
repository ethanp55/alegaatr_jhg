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

# Train neural networks for each generator
for generator_idx, x in generator_to_alignment_vectors.items():
    # Get training and validation data
    y = generator_to_correction_terms[generator_idx].reshape(-1, 1)
    training_data = np.hstack((x, y))
    np.random.shuffle(training_data)
    train_cutoff_index = int(len(training_data) * 0.75)
    train_set, validation_set = training_data[:train_cutoff_index], training_data[train_cutoff_index:]
    x_train, y_train, x_validation, y_validation = \
        train_set[:, :-1], train_set[:, -1], validation_set[:, :-1], validation_set[:, -1]

    # Scale the input data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)

    # Try different hyperparameters and pick the best set (cross validation with 5 folds)
    print(f'Training generator {generator_idx}')
    print('X train shape: ' + str(x_train_scaled.shape))
    print('Y train shape: ' + str(y_train.shape))
    param_grid = {
        'hidden_layer_sizes': [(25, 50), (50, 100), (25, 25), (50, 50), (100, 100), (25, 50, 25), (100, 150, 100)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
    }
    random_search = RandomizedSearchCV(MLPRegressor(), param_grid, cv=5, scoring='r2', n_iter=10)
    random_search.fit(x_train_scaled, y_train)
    mlp = random_search.best_estimator_
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best R-squared: {random_search.best_score_}')
    print(
        f'Best validation R-squared: {r2_score(y_validation, mlp.predict(x_validation_scaled))}\n')

    # Store the best model and the scaler
    with open(f'../aat/nn_models/generator_{generator_idx}_nn.pickle', 'wb') as f:
        pickle.dump(mlp, f)

    with open(f'../aat/nn_models/generator_{generator_idx}_scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)
