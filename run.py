import math
import sys

import numpy as np
import pandas as pd
from IPython import display

from helper import tf_basic_model

pd.options.display.max_rows = 100
pd.options.display.max_columns = 200
pd.options.display.float_format = '{:.1f}'.format

housing_data_frame = pd.read_csv('./data/train.csv', sep=",")
testing_housing_data_frame = pd.read_csv('./data/test.csv', sep=",")
housing_data_frame = housing_data_frame.reindex(np.random.permutation(housing_data_frame.index))
# display.display(housing_data_frame.describe(include = 'all'))

training_dataframe = housing_data_frame.head(1260)
validation_dataframe = housing_data_frame.tail(200)


training_examples = tf_basic_model.preprocess_features(training_dataframe)
# display.display(training_examples.dtypes)
# # display.display(training_examples)

training_targets = tf_basic_model.preprocess_targets(training_dataframe)
validation_examples = tf_basic_model.preprocess_features(validation_dataframe)
validation_targets = tf_basic_model.preprocess_targets(validation_dataframe)
testing_examples = tf_basic_model.preprocess_features(testing_housing_data_frame)

# print('Training examples summary:')
# display.display(training_examples.describe())
# print('Validation examples summary:')
# display.display(validation_examples.describe())

linear_regressor = tf_basic_model.train_model(learning_rate=0.003,
                                              steps=500,
                                              batch_size=100,
                                              feature_columns=tf_basic_model.construct_feature_columns(training_examples),
                                              training_examples=training_examples,
                                              training_targets=training_targets,
                                              validation_examples=validation_examples,
                                              validation_targets=validation_targets)

prediction = tf_basic_model.submit_prediction(linear_regressor, testing_examples)
