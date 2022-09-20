import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPool2D, Dense,
    Dropout)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import warnings
import numpy as np


# %% Data

no_of_train_img = 3000
no_of_test_img = 1000

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = tf.reshape(X_train, (-1, 28, 28, 1)) / 255
X_test = tf.reshape(X_test, (-1, 28, 28, 1)) / 255
X_train = np.take(X_train, list(range(no_of_train_img)), axis=0)
y_train = np.take(y_train, list(range(no_of_train_img)), axis=0)
X_test = np.take(X_test, list(range(no_of_test_img)), axis=0)
y_test = np.take(y_test, list(range(no_of_test_img)), axis=0)


# %% Model

def get_model(input_shape, no_classes):
    inputs = Input(shape=input_shape)
    initializer = tf.keras.initializers.HeNormal()
    x = Conv2D(32, (3, 3), padding="same", activation="relu",
               kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu",
               kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu",
               kernel_initializer=initializer)(x)
    x = GlobalMaxPool2D()(x)
    x = Dropout(0.4)(x, training=False)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    outputs = Dense(no_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# %% Training

model1 = get_model((28, 28, 1), 10)
model1.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),
    loss=SparseCategoricalCrossentropy(from_logits=True))

cb1 = callbacks.EarlyStopping(
    monitor="val_loss", mode="min", verbose=0, patience=7)
cb2 = callbacks.ModelCheckpoint(
    "./model", monitor="val_loss", save_best_only=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    history1 = model1.fit(X_train, y_train, epochs=10, batch_size=32,
                          validation_data=(X_test, y_test), verbose=1,
                          callbacks=[cb1, cb2], shuffle=True)

model1 = tf.keras.models.load_model("./model", compile=False)
pred_classes1 = np.argmax(model1.predict(X_test), axis=1)
scorer_accuracy = tf.keras.metrics.Accuracy()
accuracy_m1 = scorer_accuracy(y_test, pred_classes1)
print(f"Accuracy on the validation data: {accuracy_m1:.3f}")
