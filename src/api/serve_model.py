import logging

import numpy as np
import tensorflow as tf

from helpers import constants

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()

EPOCHS = 20


def get_model_api():
    """Returns lambda function for api"""
    # Load model
    model = tf.keras.models.load_model(constants.TF_MODEL_PATH)

    def model_api(input_data):
        # 2. process input
        input_data_processed = np.array([input_data])

        # 3. call model predict function
        preds = model.predict(input_data_processed)

        # 4. process the output
        output_data = preds[0].tolist()

        # 5. return the output for the api
        return output_data

    return model_api
