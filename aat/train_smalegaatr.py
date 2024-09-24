from GeneSimulation_py.smalegaatr import SingleGenModel
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.optimizers import Adam

ENHANCED = True
N_EPOCHS = 500
VALIDATION_PERCENTAGE = 0.3
EARLY_STOP = int(N_EPOCHS * 0.1)
BATCH_SIZE = 512

alignment_vectors, states, labels = None, None, None
training_data_folder = '../aat/training_data/'
adjustment = '_enh' if ENHANCED else ''

for file in os.listdir(training_data_folder):
    if ('sin_c' not in file) or (ENHANCED and '_enh' not in file) or (not ENHANCED and '_enh' in file):
        continue

    data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)
    is_alignment_vectors = 'vectors' in file
    is_state = 'states' in file

    if is_alignment_vectors:
        if alignment_vectors is None:
            alignment_vectors = data

        else:
            alignment_vectors = np.concatenate([alignment_vectors, data])

    elif is_state:
        if states is None:
            states = data

        else:
            states = np.concatenate([states, data])

    else:
        if labels is None:
            labels = data

        else:
            labels = np.concatenate([labels, data])

# Make sure we read the data in properly
assert alignment_vectors.shape[0] == states.shape[0] == labels.shape[0]

# Create training and validation sets
n_samples = alignment_vectors.shape[0]
n_val = int(n_samples * VALIDATION_PERCENTAGE)
indices = np.arange(n_samples)
np.random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]
assert len(val_indices) == n_val
assert len(train_indices) == n_samples - n_val
aat_vec_train, states_train, y_train = \
    alignment_vectors[train_indices, :], states[train_indices, :], labels[train_indices,]
aat_vec_val, states_val, y_val = \
    alignment_vectors[val_indices, :], states[val_indices, :], labels[val_indices,]
assert aat_vec_train.shape[0] == states_train.shape[0] == y_train.shape[0]
assert aat_vec_val.shape[0] == states_val.shape[0] == y_val.shape[0]
print(aat_vec_train.shape, states_train.shape, y_train.shape)
print(aat_vec_val.shape, states_val.shape, y_val.shape)

# State scaler
scaler = StandardScaler()
states_train = scaler.fit_transform(states_train)
states_val = scaler.transform(states_val)
with open(f'../aat/single_gen_model/single_gen_scaler{adjustment}.pickle', 'wb') as f:
    pickle.dump(scaler, f)

# Create the model
state_dim = (30 ** 2) + (2 * 30)
assert aat_vec_train.shape[1] == 59
assert states_train.shape[1] == state_dim
model = SingleGenModel(59, state_dim, 8)

# Items needed for training
optimizer = Adam()
loss_fn = MeanSquaredError()
val_metric = MeanSquaredErrorMetric()
test_metric = MeanSquaredErrorMetric()
best_val_mse = np.inf
n_epochs_without_change = 0

# Lists to monitor training and validation losses over time - used to generate a plot at the end of training
training_losses, validation_losses = [], []

# Batches
n_batches, n_val_batches = len(aat_vec_train) // BATCH_SIZE, len(aat_vec_val) // BATCH_SIZE

for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')

    # Iterate through training batches, tune model
    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        curr_aat_vec, curr_states, curr_y = \
            aat_vec_train[start_idx:end_idx, :], states_train[start_idx:end_idx, :], y_train[start_idx:end_idx, ]

        with tf.GradientTape() as tape:
            predictions = model((curr_aat_vec, curr_states), training=True)
            loss = loss_fn(curr_y, predictions)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Once all batches for the epoch are complete, calculate the validation loss
    for i in range(n_val_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        curr_aat_vec, curr_states, curr_y = \
            aat_vec_val[start_idx:end_idx, :], states_val[start_idx:end_idx, :], y_val[start_idx:end_idx, ]

        val_predictions = model((curr_aat_vec, curr_states))
        val_metric.update_state(curr_y, val_predictions)

    val_mse = val_metric.result()
    val_metric.reset_state()

    print(f'Train MSE = {loss}, Validation MSE = {val_mse}')

    # Add the training and validation losses to their corresponding lists
    training_losses, validation_losses = training_losses + [loss], validation_losses + [val_mse]

    # Make updates if the validation performance improved
    if val_mse < best_val_mse:
        print(f'Validation performance improved from {best_val_mse} to {val_mse}')

        # Reset the number of epochs that have passed without any improvement/change
        n_epochs_without_change = 0

        # Update the best validation performance metric
        best_val_mse = val_mse

        # Save the network
        model.save(f'../aat/single_gen_model/single_gen_model{adjustment}.keras')

    else:
        # Increment the number of epochs that have passed without any improvement/change
        n_epochs_without_change += 1

        # If sufficient epochs have passed without improvement, cancel the training process
        if n_epochs_without_change >= EARLY_STOP:
            print(f'EARLY STOPPING - {n_epochs_without_change} HAVE PASSED WITHOUT VALIDATION IMPROVEMENT')
            break

    # Shuffle the training data at the end of each epoch
    indices = np.arange(len(aat_vec_train))
    np.random.shuffle(indices)
    aat_vec_train, states_train, y_train = aat_vec_train[indices, :], states_train[indices, :], y_train[indices,]
