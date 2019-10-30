import pandas as pd
import logging

from helpers import constants

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()


def main():

    logger.info('Data preparation')

    # Read CSV Data with no header
    logger.info('Reading data from {} ...'.format(constants.ORIGINAL_DATA_FILE))
    data = pd.read_csv(constants.ORIGINAL_DATA_FILE, header=None)
    logger.info('Data read. Shape: {}'.format(data.shape))

    # Remove na's in the data
    logger.info('Removing NA values from data ...')
    data.dropna(inplace=True)

    # Se see that there is a ')' at the end of the field. Let's remove it
    logger.info('Removing extra parenthesis ...')
    data[3] = data[3].str.replace(')', '')

    # Let's expand the 3rd field into separate features
    logger.info('Creating features ...')
    features = data[3].str.split("-", expand=True)

    # Let's compute the true value of the completion rate
    logger.info('Computing completion rate ...')
    completion_rate = pd.DataFrame(data[2] / data[1])

    # Join the data
    logger.info('Saving clean data to {}'.format(constants.CLEAN_DATA_FILE))
    data_clean = pd.concat([features, completion_rate], axis=1)
    data_clean.to_csv(constants.CLEAN_DATA_FILE, index=False)


if __name__ == "__main__":
    main()
