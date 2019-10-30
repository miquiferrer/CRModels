import logging
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from helpers.datasets import load_training_test_datasets
from helpers import constants

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()


def main():

    logger.info('ScikitLaern Linear Regression Model')

    # Read CSV Data with no header
    logger.info('Loading training ans test data ...')
    X_train, X_test, y_train, y_test = load_training_test_datasets(constants.XTRAIN_FILE, constants.XTEST_FILE, constants.YTRAIN_FILE, constants.YTEST_FILE)
    print(X_train.shape)

    # Load model
    logger.info('Loading model ...')
    regressor = LinearRegression()

    # Train model
    logger.info('Training model ...')
    regressor.fit(X_train, y_train.values)  # training the algorithm

    # Make predictions
    logger.info('Making predictions ...')
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    logger.info('Model evaluation:')
    logger.info(' * Mean Absolute Error: {}'.format(mean_absolute_error(y_test, y_pred)))
    logger.info(' * Mean Squared Error: {}'.format(mean_squared_error(y_test, y_pred)))
    logger.info(' * Root Mean Squared Error: {}'. format(np.sqrt(mean_squared_error(y_test, y_pred))))


if __name__ == "__main__":
    main()

