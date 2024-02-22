import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate dummy data
np.random.seed(0)
x_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=(1000, 1))

x_val = np.random.rand(200, 10)
y_val = np.random.randint(2, size=(200, 1))

# Define the model with dropout layers
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),  # Dropout layer with a dropout rate of 0.5
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),  # Dropout layer with a dropout rate of 0.5
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val)
)
