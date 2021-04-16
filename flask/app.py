#
# Copyright (c) 2020 Cord Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# External imports
import cv2
import base64
import numpy as np
import logging
import time

from logging.config import dictConfig
from flask import Flask, request, make_response


PORT = 8080

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

logging.info('Version 1.0')

app = Flask(__name__)


@app.route('/detection/predict', methods=['POST'])
def predict():
    """
    Object detection inference endpoint for single images.

    Takes an image binary and runs inference.
    Returns prediction dict in Cord format.

    Submit POST request to http://0.0.0.0:8080/detection/predict

    Served via production WSGI Waitress web server.

    Example:
    ::
        detector = ObjectDetector(0.8, 0.3)
        prediction = detector.predict(frame)
    """
    # Buffer image
    image_binary = request.files['image_binary'].read()
    nparr = np.frombuffer(image_binary, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_binary is not None:

        # Run predictions
        try:
            prediction = {}
        except Exception as e:
            logging.error(e)
            return make_response(generate_success_response("Inference failed"))

        return make_response(generate_success_response(prediction))
    else:
        return make_response(generate_error_response("Did not receive frame"))


@app.after_request
def apply_caching(response):
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'POST')
    response.headers.set('Access-Control-Allow-Headers', '*')
    response.headers.set('Access-Control-Allow-Credentials', 'true')
    response.headers.set("Accept-Encoding", "gzip")
    return response


# ---------------------------------------------------------------
#                  Helper functions
# ---------------------------------------------------------------
def generate_success_response(data):
    return {"status": True,
            "response": data}


def generate_error_response(data):
    return {"status": False,
            "response": data}


if __name__ == '__main__':
    from waitress import serve

    serve(app, host="0.0.0.0", port=PORT)
