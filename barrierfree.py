import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from barrier_free_model import BFModel, process_img, decode_predictions


MODEL_PATH = '0507-gap-densenet201-size299-half-noise-N3-acc-017-0.8703.h5'
TARGET_SIZE = (299, 299)

app = Flask(__name__)
CORS(app)

model = None


def load_defined_model():

    global model
    model = BFModel(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def image_classifier():

    image = request.files['image'].read()
    image = io.BytesIO(image)
    image = Image.open(image)
    image = process_img(image, TARGET_SIZE)

    pred = model.predict(image)
    result, y_pred = decode_predictions(pred)

    result_dict = {'result': []}

    for r in result:
        result_dict['result'].append(r)

    return jsonify(result_dict)


if __name__ == "__main__":
    print('* Loading model')
    load_defined_model()
    print('* Starting server')
    app.run()
