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

# L2 Regularization
model_l2 = keras.Sequential(
    [
        layers.Dense(
            64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        layers.Dense(
            64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Dropout
model_dropout = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, min_delta=0.001, restore_best_weights=True
)

# Compile the models
model_l2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model_dropout.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Train the models
history_l2 = model_l2.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

history_dropout = model_dropout.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)
