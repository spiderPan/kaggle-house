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
        selected_features = housing_data_frame.copy()
        not_features_indexes = pd.Index(['SalePrice'])
        preprocess_features_index = selected_features.columns.difference(not_features_indexes)
        preprocess_features = selected_features[preprocess_features_index]

        int_cols = preprocess_features.select_dtypes(include=['int64', 'float64']).columns.drop('Id')
        obj_cols = preprocess_features.select_dtypes(include=['object']).columns

        preprocess_features[int_cols] = preprocess_features[int_cols].fillna(0)
        preprocess_features[obj_cols] = preprocess_features[obj_cols].fillna('NONE')

        return preprocess_features

    def preprocess_targets(housing_data_frame):
        output_targets = pd.DataFrame()
        house_sale_price = housing_data_frame.get('SalePrice', 0)
        if house_sale_price is 0:
            return pd.DataFrame(0, index=np.arange(len(housing_data_frame)), columns=['SalePrice'])

        output_targets['SalePrice'] = house_sale_price
        return output_targets

    def construct_feature_columns(input_features):
        engineered_features = []
        int_cols = input_features.select_dtypes(include=['int64', 'float64']).columns.drop('Id')
        obj_cols = input_features.select_dtypes(include=['object']).columns

        for continuous_feature in list(int_cols):
            engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))

        for categorical_feature in list(obj_cols):
            sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)
            engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16, combiner="sum"))

        return engineered_features

    def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
        features = {key: np.array(value) for key, value in dict(features).items()}

        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def train_nn_regression_model(
            my_optimizer,
            steps,
            batch_size,
            hidden_units,
            training_examples,
            training_targets,
            validation_examples,
            validation_targets):

        periods = 100
        steps_per_period = steps / periods

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

        return dnn_regressor, training_rmse, validation_rmse

    def get_input_fn(data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(x=pd.DataFrame({k: data_set[k].values for k in data_set.columns}), y=None, num_epochs=num_epochs, shuffle=shuffle)

    def submit_prediction(model, testing_examples, testing_targets, filename=None):
        if filename is None:
            filename = 'submission'

        def predict_testing_input_fn(): return tf_basic_model.my_input_fn(testing_examples, testing_targets['SalePrice'], num_epochs=1, shuffle=False)
        submission = pd.DataFrame()
        submission['Id'] = testing_examples['Id']
        predictions = model.predict(input_fn=predict_testing_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        submission['SalePrice'] = predictions
        submission.head()
        submission.to_csv('./data/' + filename + '.csv', index=False)
