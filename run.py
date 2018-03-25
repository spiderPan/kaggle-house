import math

import numpy as np
import pandas as pd
from IPython import display

housing_data_frame = pd.read_csv('./data/train.csv', sep=",")
housing_data_frame = housing_data_frame.reindex(np.random.permutation(housing_data_frame.index))
display.display(housing_data_frame.describe())

#
# training_dataframe = housing_data_frame.head(12000)
# validation_dataframe = housing_data_frame.tail(5000)
# training_examples = preprocess_features(training_dataframe)
# training_targets = preprocess_targets(training_dataframe)
# validation_examples = preprocess_features(validation_dataframe)
# validation_targets = preprocess_targets(validation_dataframe)
#
# print('Training examples summary:')
# display.display(training_examples.describe())
# print('Validation examples summary:')
# display.display(validation_examples.describe())
#
# print('Training targets summary:')
# display.display(training_targets.describe())
# print('Validation targets summary:')
# display.display(validation_targets.describe())
#
# correlation_dataframe = training_examples.copy()
# correlation_dataframe['target'] = training_targets['median_house_value']
#
# display.display(correlation_dataframe.corr())
# plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
#
# selected_training_examples = select_and_transform_features(training_examples)
# selected_validation_examples = select_and_transform_features(validation_examples)
#
# minimal_features = [
#     'median_income',
# ]
#
#
# assert minimal_features, "You must select at least one feature!"
#
# minimal_training_examples = training_examples[minimal_features]
# minimal_validation_examples = validation_examples[minimal_features]
#
# train_model(
#     learning_rate=0.01,
#     steps=500,
#     batch_size=5,
#     training_examples=selected_training_examples,
#     training_targets=training_targets,
#     validation_examples=selected_validation_examples,
#     validation_targets=validation_targets,
# )
