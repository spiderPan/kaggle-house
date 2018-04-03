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
        house_sale_price = housing_data_frame.get('SalePrice',0)
        if house_sale_price is 0:
            return pd.DataFrame(0, index=np.arange(len(housing_data_frame)), columns=['SalePrice'])

        output_targets['SalePrice'] = house_sale_price
        return output_targets

    def construct_feature_columns(input_features):
        LotFrontage = tf.feature_column.numeric_column('LotFrontage')
        MasVnrArea = tf.feature_column.numeric_column('MasVnrArea')
        GarageYrBlt = tf.feature_column.numeric_column('GarageYrBlt')
        MSSubClass = tf.feature_column.numeric_column('MSSubClass')

        LotArea = tf.feature_column.numeric_column('LotArea')
        OverallQual = tf.feature_column.numeric_column('OverallQual')
        OverallCond = tf.feature_column.numeric_column('OverallCond')
        YearBuilt = tf.feature_column.numeric_column('YearBuilt')

        YearRemodAdd = tf.feature_column.numeric_column('YearRemodAdd')
        BsmtFinSF1 = tf.feature_column.numeric_column('BsmtFinSF1')
        BsmtFinSF2 = tf.feature_column.numeric_column('BsmtFinSF2')
        BsmtUnfSF = tf.feature_column.numeric_column('BsmtUnfSF')
        TotalBsmtSF = tf.feature_column.numeric_column('TotalBsmtSF')

        firstFlrSF = tf.feature_column.numeric_column('1stFlrSF')
        secondFlrSF = tf.feature_column.numeric_column('2ndFlrSF')
        LowQualFinSF = tf.feature_column.numeric_column('LowQualFinSF')
        GrLivArea = tf.feature_column.numeric_column('GrLivArea')
        BsmtFullBath = tf.feature_column.numeric_column('BsmtFullBath')

        BsmtHalfBath = tf.feature_column.numeric_column('BsmtHalfBath')
        FullBath = tf.feature_column.numeric_column('FullBath')
        HalfBath = tf.feature_column.numeric_column('HalfBath')
        BedroomAbvGr = tf.feature_column.numeric_column('BedroomAbvGr')
        KitchenAbvGr = tf.feature_column.numeric_column('KitchenAbvGr')

        TotRmsAbvGrd = tf.feature_column.numeric_column('TotRmsAbvGrd')
        Fireplaces = tf.feature_column.numeric_column('Fireplaces')
        GarageCars = tf.feature_column.numeric_column('GarageCars')
        GarageArea = tf.feature_column.numeric_column('GarageArea')
        WoodDeckSF = tf.feature_column.numeric_column('WoodDeckSF')

        OpenPorchSF = tf.feature_column.numeric_column('OpenPorchSF')
        EnclosedPorch = tf.feature_column.numeric_column('EnclosedPorch')
        SsnPorch = tf.feature_column.numeric_column('3SsnPorch')
        ScreenPorch = tf.feature_column.numeric_column('ScreenPorch')
        PoolArea = tf.feature_column.numeric_column('PoolArea')

        MiscVal = tf.feature_column.numeric_column('MiscVal')
        MoSold = tf.feature_column.numeric_column('MoSold')
        YrSold = tf.feature_column.numeric_column('YrSold')

        MSZoning = tf.feature_column.categorical_column_with_hash_bucket('MSZoning', hash_bucket_size=1000)
        Street = tf.feature_column.categorical_column_with_hash_bucket('Street', hash_bucket_size=1000)
        LotShape = tf.feature_column.categorical_column_with_hash_bucket('LotShape', hash_bucket_size=1000)
        LandContour = tf.feature_column.categorical_column_with_hash_bucket('LandContour', hash_bucket_size=1000)
        Utilities = tf.feature_column.categorical_column_with_hash_bucket('Utilities', hash_bucket_size=1000)

        LotConfig = tf.feature_column.categorical_column_with_hash_bucket('LotConfig', hash_bucket_size=1000)
        LandSlope = tf.feature_column.categorical_column_with_hash_bucket('LandSlope', hash_bucket_size=1000)
        Neighborhood = tf.feature_column.categorical_column_with_hash_bucket('Neighborhood', hash_bucket_size=1000)
        Condition1 = tf.feature_column.categorical_column_with_hash_bucket('Condition1', hash_bucket_size=1000)
        Condition2 = tf.feature_column.categorical_column_with_hash_bucket('Condition2', hash_bucket_size=1000)

        BldgType = tf.feature_column.categorical_column_with_hash_bucket('BldgType', hash_bucket_size=1000)
        HouseStyle = tf.feature_column.categorical_column_with_hash_bucket('HouseStyle', hash_bucket_size=1000)
        RoofStyle = tf.feature_column.categorical_column_with_hash_bucket('RoofStyle', hash_bucket_size=1000)
        RoofMatl = tf.feature_column.categorical_column_with_hash_bucket('RoofMatl', hash_bucket_size=1000)
        Exterior1st = tf.feature_column.categorical_column_with_hash_bucket('Exterior1st', hash_bucket_size=1000)

        Exterior2nd = tf.feature_column.categorical_column_with_hash_bucket('Exterior2nd', hash_bucket_size=1000)
        MasVnrType = tf.feature_column.categorical_column_with_hash_bucket('MasVnrType', hash_bucket_size=1000)
        ExterQual = tf.feature_column.categorical_column_with_hash_bucket('ExterQual', hash_bucket_size=1000)
        ExterCond = tf.feature_column.categorical_column_with_hash_bucket('ExterCond', hash_bucket_size=1000)
        Foundation = tf.feature_column.categorical_column_with_hash_bucket('Foundation', hash_bucket_size=1000)

        BsmtQual = tf.feature_column.categorical_column_with_hash_bucket('BsmtQual', hash_bucket_size=1000)
        BsmtCond = tf.feature_column.categorical_column_with_hash_bucket('BsmtCond', hash_bucket_size=1000)
        BsmtExposure = tf.feature_column.categorical_column_with_hash_bucket('BsmtExposure', hash_bucket_size=1000)
        BsmtFinType1 = tf.feature_column.categorical_column_with_hash_bucket('BsmtFinType1', hash_bucket_size=1000)
        BsmtFinType2 = tf.feature_column.categorical_column_with_hash_bucket('BsmtFinType2', hash_bucket_size=1000)

        Heating = tf.feature_column.categorical_column_with_hash_bucket('Heating', hash_bucket_size=1000)
        HeatingQC = tf.feature_column.categorical_column_with_hash_bucket('HeatingQC', hash_bucket_size=1000)
        CentralAir = tf.feature_column.categorical_column_with_hash_bucket('CentralAir', hash_bucket_size=1000)
        Electrical = tf.feature_column.categorical_column_with_hash_bucket('Electrical', hash_bucket_size=1000)
        KitchenQual = tf.feature_column.categorical_column_with_hash_bucket('KitchenQual', hash_bucket_size=1000)

        Functional = tf.feature_column.categorical_column_with_hash_bucket('Functional', hash_bucket_size=1000)
        FireplaceQu = tf.feature_column.categorical_column_with_hash_bucket('FireplaceQu', hash_bucket_size=1000)
        GarageType = tf.feature_column.categorical_column_with_hash_bucket('GarageType', hash_bucket_size=1000)
        GarageFinish = tf.feature_column.categorical_column_with_hash_bucket('GarageFinish', hash_bucket_size=1000)
        GarageQual = tf.feature_column.categorical_column_with_hash_bucket('GarageQual', hash_bucket_size=1000)

        GarageCond = tf.feature_column.categorical_column_with_hash_bucket('GarageCond', hash_bucket_size=1000)
        PavedDrive = tf.feature_column.categorical_column_with_hash_bucket('PavedDrive', hash_bucket_size=1000)
        SaleType = tf.feature_column.categorical_column_with_hash_bucket('SaleType', hash_bucket_size=1000)
        SaleCondition = tf.feature_column.categorical_column_with_hash_bucket('SaleCondition', hash_bucket_size=1000)
        #Alley = tf.feature_column.categorical_column_with_hash_bucket('Alley', hash_bucket_size=1000)
        #PoolQC = tf.feature_column.categorical_column_with_hash_bucket('PoolQC', hash_bucket_size=1000)
        #Fence = tf.feature_column.categorical_column_with_hash_bucket('Fence', hash_bucket_size=1000)
        #MiscFeature = tf.feature_column.categorical_column_with_hash_bucket('MiscFeature', hash_bucket_size=1000)

        feature_columns = set([
            MSSubClass, MSZoning, LotFrontage, LotArea, Street, LotShape, LandContour, Utilities, LotConfig,
            LandSlope, Neighborhood, Condition1, Condition2, BldgType,
            HouseStyle, OverallQual, OverallCond, YearBuilt, YearRemodAdd,
            RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType,
            MasVnrArea, ExterQual, ExterCond, Foundation, BsmtQual,
            BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1,
            BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, Heating,
            HeatingQC, CentralAir, Electrical, firstFlrSF, secondFlrSF,
            LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath,
            HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual,
            TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageType,
            GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual,
            GarageCond, PavedDrive, WoodDeckSF, OpenPorchSF,
            EnclosedPorch, SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold, SaleType,
            SaleCondition])
        # Alley, PoolQC, Fence, MiscFeature])

        return feature_columns

    def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
        features = {key: np.array(value) for key, value in dict(features).items()}

        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def train_model(learning_rate, steps, batch_size, feature_columns, training_examples, training_targets, validation_examples, validation_targets):
        periods = 20
        steps_per_period = steps / periods

        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

        def training_input_fn(): return tf_basic_model.my_input_fn(training_examples, training_targets['SalePrice'], batch_size=batch_size)

        def predict_training_input_fn(): return tf_basic_model.my_input_fn(training_examples, training_targets['SalePrice'], num_epochs=1, shuffle=False)

        def predict_validation_input_fn(): return tf_basic_model.my_input_fn(validation_examples, validation_targets['SalePrice'], num_epochs=1, shuffle=False)

        print('Training model...')
        print('RMSE (on training data): ')
        training_rmse = []
        validation_rmse = []

        for period in range(0, periods):
            linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
            training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
            training_predictions = np.array([item['predictions'][0] for item in training_predictions])

            validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

            training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
            validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
            print('period % 02d: % 0.2f' % (period, training_root_mean_squared_error))
            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)
        print('Model training finished')
        return linear_regressor

    def get_input_fn(data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(x=pd.DataFrame({k: data_set[k].values for k in data_set.columns}), y=None,num_epochs=num_epochs,shuffle=shuffle)

    def submit_prediction(model, testing_examples, testing_targets):
        def predict_testing_input_fn(): return tf_basic_model.my_input_fn(testing_examples, testing_targets['SalePrice'], num_epochs=1, shuffle=False)
        submission = pd.DataFrame()
        submission['Id'] = testing_examples['Id']
        predictions = model.predict(input_fn=predict_testing_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        submission['SalePrice'] = predictions
        submission.head()
        submission.to_csv('./data/submission.csv', index=False)
