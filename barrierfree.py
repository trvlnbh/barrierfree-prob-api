import io
import argparse
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from barrier_free_model import BFModel, process_img, decode_predictions


# Enter a model and a input size tuple to use for ensemble
MODEL_PATH = '0507-gap-densenet201-size299-half-noise-N3-acc-017-0.8703.h5'
TARGET_SIZE = (299, 299)

# Enter a list of models and input size tuples to use for ensemble in order
ENSEMBLE_LIST = []
TARGET_SIZE_LIST = []
NUM_ENSEMBLE = None

app = Flask(__name__)
CORS(app)

is_ensemble = None
model = None


def load_defined_model():

    global model
    print('* Loading model')

    if is_ensemble:
        print('* Ensemble mode')
        model = []
        for m in ENSEMBLE_LIST:
            model.append(BFModel(m))
    else:
        print('* Single mode')
        model = BFModel(MODEL_PATH)

    print('* Model is loaded')


@app.route('/predict', methods=['POST'])
def image_classifier():

    image = request.files['image'].read()
    image = io.BytesIO(image)
    image = Image.open(image)

    if is_ensemble:
        pred = np.zeros(shape=(1, 2))
        for idx, ts in enumerate(TARGET_SIZE_LIST):
            _image = process_img(image, ts)
            pred += model[idx].predict(_image)
        pred /= NUM_ENSEMBLE
    else:
        image = process_img(image, TARGET_SIZE)
        pred = model.predict(image)

    result, y_pred = decode_predictions(pred)

    result_dict = {'result': []}

    for r in result:
        result_dict['result'].append(r)

    return jsonify(result_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',
                        dest='predict_mode',
                        type=int,
                        default=1,
                        help='Predict using single model or average ensemble')
    args = parser.parse_args()

    if args.predict_mode == 1:
        is_ensemble = False
    elif args.predict_mode == 2:
        is_ensemble = True
    else:
        raise argparse.ArgumentTypeError('Mode value must be 1 or 2')
    load_defined_model()

    print('* Starting server')
    app.run()
