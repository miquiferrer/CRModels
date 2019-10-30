import pandas as pd
import logging

from sklearn.model_selection import train_test_split

from helpers.datasets import save_dataframe
from helpers import constants

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()

TEST_RATIO = 0.2
NUM_RECORDS_TO_KEEP = 100000


def main():

    logger.info('Generate training and tests sets')

    # Read CSV Data with no header
    logger.info('Reading data from {} ...'.format(constants.CLEAN_DATA_FILE))
    data = pd.read_csv(constants.CLEAN_DATA_FILE, header=0)
    data.dropna(inplace=True)
    logger.info('Data read. Shape: {}'.format(data.shape))

    # Generating training and test sets (for demo purposes we only keep 100k random
    logger.info('Generating training and test sets ...')
    ncols = data.shape[1]
    features = data[data.columns[0:ncols-1]]
    completion_rate = data['0.1']

    # Build training and tests sets (test set = 20% of the data)
    X_train, X_test, y_train, y_test = train_test_split(features, completion_rate, test_size=TEST_RATIO, random_state=0)
    training_samples = int(NUM_RECORDS_TO_KEEP*(1-TEST_RATIO))
    test_samples = int(NUM_RECORDS_TO_KEEP*TEST_RATIO)

    # Join the data
    logger.info('Saving training and test sets ...')
    logger.info(' * Training samples: {}'.format(training_samples))
    logger.info(' * Test samples: {}'.format(test_samples))
    save_dataframe(X_train[0:training_samples], constants.XTRAIN_FILE)
    save_dataframe(X_test[0:test_samples], constants.XTEST_FILE)
    save_dataframe(pd.DataFrame(y_train)[0:training_samples], constants.YTRAIN_FILE)
    save_dataframe(pd.DataFrame(y_test)[0:test_samples], constants.YTEST_FILE)


if __name__ == "__main__":
    main()
