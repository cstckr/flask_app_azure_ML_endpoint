import json
import numpy as np


def PIL_image_to_json(img):
    return json.dumps(np.array(img).tolist())
