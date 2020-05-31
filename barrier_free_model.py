from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


def BFModel(model_path):

    model = load_model(model_path)

    return model


def process_img(image, target_size):

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.

    return image


def decode_predictions(prediction):

    acc = ('Accessible', float(prediction[0][0]))
    inacc = ('Inaccessible', float(prediction[0][1]))

    dict_arr = [acc, inacc]
    pred_class = np.argmax(prediction[0])

    return dict_arr, pred_class
