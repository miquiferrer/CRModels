import logging
import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from helpers.datasets import load_training_test_datasets
from helpers import constants

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()


def main():

    logger.info('XGBoost Regression Model')

    # Read CSV Data with no header
    logger.info('Loading training ans test data ...')
    X_train, X_test, y_train, y_test = load_training_test_datasets(constants.XTRAIN_FILE, constants.XTEST_FILE, constants.YTRAIN_FILE, constants.YTEST_FILE)

    # Load model
    logger.info('Loading model ...')
    # Define model
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                              colsample_bytree=0.3,
                              learning_rate=0.1,
                              max_depth=5,
                              alpha=10,
                              n_estimators=10)

    # Train model
    logger.info('Training model ...')
    xg_reg.fit(X_train.values, y_train.values)

    # Make predictions
    logger.info('Making predictions ...')
    y_pred = xg_reg.predict(X_test.values)

    # Evaluate the model
    logger.info('Model evaluation:')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info('RMSE: {}'.format(rmse))

    # Perform cross validation
    logger.info('Performing cross validation ...')
    params = {"objective":"reg:squarederror",
              'colsample_bytree': 0.3,
              'learning_rate': 0.1,
              'max_depth': 5,
              'alpha': 10}

    data_dmatrix = xgb.DMatrix(data=pd.concat([X_train, X_test]).values, label=pd.concat([y_train, y_test]).values)

    cv_results = xgb.cv(dtrain=data_dmatrix,
                        params=params,
                        nfold=3,
                        num_boost_round=50,
                        early_stopping_rounds=10,
                        metrics="rmse",
                        as_pandas=True,
                        seed=123)

    # Display results
    logger.info('Model evaluation:')
    logger.info('RMSE: {}'.format(cv_results["test-rmse-mean"].tail(1)))


if __name__ == "__main__":
    main()

