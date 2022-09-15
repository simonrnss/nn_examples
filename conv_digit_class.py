# %% Imprts
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import pylab as plt
%matplotlib inline

# %% Load dataset and plot some digits
(trainX, trainy), (testX, testy) = mnist.load_data()
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# %% Reshape data to match tensorflow requirements
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)
# %% Make y categorical
trainy = tf.keras.utils.to_categorical(trainy)
testy = tf.keras.utils.to_categorical(testy)

# %%
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255
# %%
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(3, (5, 5), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((8, 8)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# %%
history = model.fit(trainX, trainy, epochs=10, batch_size=32, validation_data=(testX, testy))

# %%
test_example = 3
plt.imshow(testX[test_example, :, :, :])
a = model.layers[0](testX[:10, :, :])
n_kern = 3
fig, ax_array = plt.subplots(1, n_kern, figsize=(15, 5))
for k in range(n_kern):
    ax_array[k].imshow(a[test_example, :, :, k])

b = model.layers[1](a)
fig, ax_array = plt.subplots(1, n_kern, figsize=(15, 5))
for k in range(n_kern):
    ax_array[k].imshow(b[test_example, :, :, k])
# %%
fig, ax_array = plt.subplots(1, n_kern, figsize=(15, 5))
for k in range(n_kern):
    ax_array[k].imshow(model.layers[0].weights[0][:, :, :, k])
# %%

import numpy as np
a = model.layers[0](testX)
b = model.layers[1](a)
c = np.array(model.layers[2](b))
zz = c[testy[:, 0] == 1, :]
for i in range(1, 10):
    zz = np.vstack((
        zz, c[testy[:, i] == 1, :]
    ))
plt.figure(figsize=(10, 10))
plt.imshow(zz, aspect='auto')
# %%
