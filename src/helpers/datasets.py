import logging

import pandas as pd

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()


def save_dataframe(df, out_file):
    # Join the data
    logger.info('Saving clean data to {}'.format(out_file))
    df.to_csv(out_file, index=False)


def load_training_test_datasets(xtrain_path, xtest_path, ytrain_path, ytest_path):
    X_train = pd.read_csv(xtrain_path, header=None)
    X_test = pd.read_csv(xtest_path, header=None)
    y_train = pd.read_csv(ytrain_path, header=None)
    y_test = pd.read_csv(ytest_path, header=None)

    return X_train, X_test, y_train, y_test
