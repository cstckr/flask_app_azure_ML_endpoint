import os
import tensorflow as tf
import numpy as np
import json


def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model")
    model = tf.keras.models.load_model(model_path, compile=False)


def run(raw_data):
    data = json.loads(raw_data)["data"]
    data = json.loads(data)
    data = np.array(data)
    data = tf.reshape(data, (-1, 28, 28, 1)) / 255
    prediction = np.argmax(model.predict(data), axis=1)[0]
    return json.dumps(prediction.tolist())
