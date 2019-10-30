import logging

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from helpers.datasets import load_training_test_datasets
from helpers import constants


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()

EPOCHS = 20


# Define function to build the model
def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


# Display training progress by printing number of epoch
class PrintEpoch(keras.callbacks.Callback):

    def __init__(self, total_epochs):
        self.epoch = 0
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs):
        self.epoch += 1

        if self.epoch % 10 == 0:
            print('Epoch {} out of {}'.format(self.epoch, self.total_epochs))


def main():

    logger.info('TensorFlow Regression Model')

    # Read CSV Data with no header
    logger.info('Loading training ans test data ...')
    X_train, X_test, y_train, y_test = load_training_test_datasets(constants.XTRAIN_FILE, constants.XTEST_FILE, constants.YTRAIN_FILE, constants.YTEST_FILE)

    # Build model
    logger.info('Build model ...')
    model = build_model(X_train.shape[1])
    logger.info(' Model summary: {}'.format(str(model.summary())))

    # Train model
    logger.info('Training model ...')
    history = model.fit(
        pd.DataFrame(X_train), pd.DataFrame(y_train),
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintEpoch(EPOCHS)])

    # Recover metrics for each epoch
    logger.info('Recovering train/validation metrics for each epoch ...')
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # Make predictions
    logger.info('Making predictions and evaluating model ...')
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
    logger.info('Testing set Mean Abs Error: {:5.2f}'.format(mae))
    logger.info('Testing set Mean Squared Error: {}'.format(mse))
    logger.info('Testing set Root Mean Squared Error: {}'.format(np.sqrt(mse)))

    # Saving model
    tf.saved_model.save(model, constants.TF_MODEL_PATH)

    # Make a prediction for the model (just for comparison with online model prediction)
    unseen_sample = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 2., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 2.]])

    prediction = model.predict(unseen_sample)
    logger.info('Model prediction for the unseen sample: {}'.format(prediction))


if __name__ == "__main__":
    main()





