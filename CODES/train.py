# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import sklearn.metrics as metrics

# Load data
train = pd.read_csv("emnist-balanced-train.csv", delimiter=',')
test = pd.read_csv("emnist-balanced-test.csv", delimiter=',')
mapp = pd.read_csv("emnist-balanced-mapping.txt", 
                   delimiter=' ', index_col=0, header=None).squeeze()

# Image size
HEIGHT = 28
WIDTH = 28

# Split features and labels
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_x = test.iloc[:, 1:]
test_y = test.iloc[:, 0]
del train, test

# Rotate images (EMNIST is rotated by default)
def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.flipud(image)
    image = np.rot90(image)
    return image

# Apply rotation
train_x = np.apply_along_axis(rotate, 1, np.asarray(train_x))
test_x = np.apply_along_axis(rotate, 1, np.asarray(test_x))

# Normalize
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

# One-hot encode labels
num_classes = train_y.nunique()
train_y = tf.keras.utils.to_categorical(train_y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

# Reshape for CNN
train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)

# Split into training and validation
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=7)

# Build CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(HEIGHT, WIDTH, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = model.fit(train_x, train_y, epochs=10, batch_size=512, verbose=1, validation_data=(val_x, val_y))

# Save model
model.save("my_model.keras")
