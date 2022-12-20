"""Simple binary classification example"""
# %%
import numpy as np
import pylab as plt
import tensorflow as tf

# %% Create a dataset
X = np.vstack(
    (
        np.random.normal(size=(50, 2)),
        np.random.normal(size=(50, 2)) + 3
    )
)
y = np.hstack((np.ones(50), np.zeros(50)))
plt.scatter(X[:, 0], X[:, 1], c=y)

# %% Create a test dataset on a grid for plotting
# This is nothing to do with neural networks, but allows us to visualise the NN
mi_x = np.floor(min(X[:, 0]))
ma_x = np.ceil(max(X[:, 0]))
mi_y = np.floor(min(X[:, 1]))
ma_y = np.ceil(max(X[:, 1]))

n_x = 200
n_y = 200
xr = np.linspace(mi_x, ma_x, n_x)
yr = np.linspace(mi_y, ma_y, n_y)

x_g, y_g = np.meshgrid(xr, yr)

test_x = np.hstack(
    (
        x_g.ravel()[:, None],
        y_g.ravel()[:, None]
    )
)

# %% Simplest possible NN. A single input layer and an output layer - equivalent to logreg
# We define: 
# - an InputLayer of size 2 (the number of dimensions of our data)
# - A single Dense layer with one neuron (our output) with a sigmoid activation (because we
#   want the output to be between 0 and 1)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2, )),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.summary() gives a nice summary of the layers
model.summary()

# %% Training
# We define an optimizer and then compile and fit the model
# When we fit, we use 20% of the data as a validation set -- this can help monitor training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=20, validation_split=0.2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.legend()

# %% Making predictions and plotting the decision boundaries that the network produces
preds = model.predict(test_x)
plt.figure(figsize=(10, 10))
plt.contour(x_g, y_g, preds.reshape((n_y, n_x)), 10)
plt.scatter(X[:, 0], X[:, 1], c=y)

# %% We can extract the trained weights...
w0 = float(model.weights[1])
w = np.array(model.weights[0])
w1 = w[0][0]
w2 = w[1][0]
print(f"w0: {w0:.2f}, w1: {w1:.2f}, w2: {w2:.2f}")

# %% Add a hidden layer with 5 neurons
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2, )),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=50, validation_split=0.2, batch_size=100)
preds = model.predict(test_x)
plt.figure(figsize=(10, 10))
plt.contour(x_g, y_g, preds.reshape((n_y, n_x)), 10)
plt.scatter(X[:, 0], X[:, 1], c=y)

# %%
import csv
with open('data.csv', 'w') as f:
    writer = csv.writer(f)
    for row in X:
        writer.writerow(row)
# %%
with open('params.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([w0, w1, w2])
# %%
