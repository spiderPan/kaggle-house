import math
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)


class tf_basic_model:
    def preprocess_features(housing_data_frame):
        int_cols = housing_data_frame.select_dtypes(include=['int64', 'float64']).columns
        obj_cols = housing_data_frame.select_dtypes(include=['object']).columns
        housing_data_frame[int_cols] = housing_data_frame[int_cols].apply(lambda x: x.astype('float64'))
        housing_data_frame[obj_cols] = housing_data_frame[obj_cols].apply(lambda x: x.astype('category').cat.codes)
        selected_features = housing_data_frame[
            [
                'Id',
                'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
                'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
                'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                'SaleCondition',
                # 'Alley', 'PoolQC', 'Fence', 'MiscFeature'
            ]
        ]
        preprocess_features = selected_features.copy()
        preprocess_features = preprocess_features.apply(lambda x: x.fillna(x.value_counts().index[0]))

        return preprocess_features

    def preprocess_targets(housing_data_frame):
        output_targets = pd.DataFrame()
        house_sale_price = housing_data_frame.get('SalePrice', 0)
        if house_sale_price is 0:
            return pd.DataFrame(0, index=np.arange(len(housing_data_frame)), columns=['SalePrice'])

        output_targets['SalePrice'] = house_sale_price
        return output_targets

    def construct_feature_columns(input_features):
        return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

    def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
        features = {key: np.array(value) for key, value in dict(features).items()}

        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def train_nn_regression_model(
            learning_rate,
            steps,
            batch_size,
            hidden_units,
            training_examples,
            training_targets,
            validation_examples,
            validation_targets):

        periods = 10
        steps_per_period = steps / periods

        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=tf_basic_model.construct_feature_columns(training_examples),
            hidden_units=hidden_units,
            optimizer=my_optimizer,
        )

        def training_input_fn(): return tf_basic_model.my_input_fn(training_examples,
                                                                   training_targets["SalePrice"],
                                                                   batch_size=batch_size)

        def predict_training_input_fn(): return tf_basic_model.my_input_fn(training_examples,
                                                                           training_targets["SalePrice"],
                                                                           num_epochs=1,
                                                                           shuffle=False)

        def predict_validation_input_fn(): return tf_basic_model.my_input_fn(validation_examples,
                                                                             validation_targets["SalePrice"],
                                                                             num_epochs=1,
                                                                             shuffle=False)

        print("Training model...")
        print("RMSE (on training data):")
        training_rmse = []
        validation_rmse = []
        for period in range(0, periods):
            dnn_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period
            )
            # Take a break and compute predictions.
            training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
            training_predictions = np.array([item['predictions'][0] for item in training_predictions])

            validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

            # Compute training and validation loss.
            training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
            # Add the loss metrics from this period to our list.
            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)
        print("Model training finished.")

        # Output a graph of loss metrics over periods.
        plt.ylabel("RMSE")
        plt.xlabel("Periods")
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(training_rmse, label="training")
        plt.plot(validation_rmse, label="validation")
        plt.legend()

        print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
        print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

        return dnn_regressor

    def get_input_fn(data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(x=pd.DataFrame({k: data_set[k].values for k in data_set.columns}), y=None, num_epochs=num_epochs, shuffle=shuffle)

    def submit_prediction(model, testing_examples, testing_targets):
        def predict_testing_input_fn(): return tf_basic_model.my_input_fn(testing_examples, testing_targets['SalePrice'], num_epochs=1, shuffle=False)
        submission = pd.DataFrame()
        submission['Id'] = testing_examples['Id']
        predictions = model.predict(input_fn=predict_testing_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        submission['SalePrice'] = predictions
        submission.head()
        submission.to_csv('./data/submission.csv', index=False)
