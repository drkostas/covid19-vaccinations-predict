import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import List, Tuple
from keras_tqdm import TQDMNotebookCallback, TQDMCallback

from data_mining import ColorizedLogger

logger = ColorizedLogger('BuildModel', 'cyan')


class BuildModel:
    __slots__ = ('window_size',)

    window_size: int

    def __init__(self, window_size: int):
        self.window_size = window_size

    def trim_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the number of weeks extra days
        num_days = df.shape[0]
        num_day_groups, extra_days = divmod(num_days, self.window_size)
        logger.info(f"Number of days in the dataset: {num_days}")
        logger.info(
            f"Number of {self.window_size}-day groups: {num_day_groups} + {extra_days} extra days")
        # Drop Last extra_days rows from the datasets
        logger.info(f"Dropping the last {extra_days} rows from the covid dataset")
        covid_df_clean_9 = df.drop(df.tail(extra_days).index)
        df = df.drop(df.tail(extra_days).index)
        logger.info(f"Dropped {num_days - covid_df_clean_9.shape[0]} rows")

        return df

    def split_dataset(self, df: pd.DataFrame, train_perc: float = 0.7) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """ Split a univariate dataset into train/test sets """

        df = df.reset_index(drop=True)
        # Calculate the train and test lengths
        total_days = len(df)
        train_days = (int(total_days * train_perc) // self.window_size) * self.window_size

        # Keep the date column and the drop
        date_train_col, date_test_col = df.date.iloc[:train_days], df.date.iloc[train_days:]
        df = df.drop(columns='date')

        # Get the values from the DF
        data = df.values

        # split into standard weeks
        train, test = data[:train_days], data[train_days:]
        # restructure into windows of weekly data
        train = np.array(np.split(train, len(train) / self.window_size))
        test = np.array(np.split(test, len(test) / self.window_size))

        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        return train, test, date_train_col, date_test_col

    @staticmethod
    def rejoin_dataset(predictions: np.ndarray, test: np.ndarray, after_pivot_columns: List[str],
                       date_column: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Flatten Numpy Arrays
        predictions_flat = predictions.reshape((predictions.shape[0] * predictions.shape[1],
                                                predictions.shape[2]))
        test_flat = test.reshape((test.shape[0] * test.shape[1], test.shape[2]))

        # Prepare for unpivoting
        after_pivot_columns_witout_date = [c for c in after_pivot_columns if c != 'date']
        date_test_col = date_column.reset_index(drop=True)

        # Convert the Numpy Arrays to Dataframes
        test_expanded_df = pd.DataFrame(test_flat, columns=after_pivot_columns_witout_date)
        predictions_expanded_df = pd.DataFrame(predictions_flat,
                                               columns=after_pivot_columns_witout_date)

        # Reinsert the date column
        test_expanded_df.insert(0, 'date', date_test_col)
        predictions_expanded_df.insert(0, 'date', date_test_col)

        return predictions_expanded_df, test_expanded_df

    @staticmethod
    def summarize_scores(name: str, score: float, scores: List[str]) -> None:
        """ Summarize scores """

        s_scores = ', '.join(['%.1f' % s for s in scores])
        logger.info('%s: [%.3f] %s' % (name, score, s_scores))

    def evaluate_model(self, train: np.ndarray, test: np.ndarray, verbose: int,
                       epochs: int, batch_size: int):
        """ Evaluate a single model """

        # fit model
        num_features = train.shape[2]
        model = self.build_model(train, self.window_size, verbose, epochs, batch_size, num_features)
        # history is a list of weekly data
        history = [x for x in train]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(test)):
            # predict the week
            yhat_sequence = self.forecast(model, history, self.window_size)
            # store the predictions
            predictions.append(yhat_sequence)
            # get real observation and add to history for predicting the next week
            history.append(test[i, :])
        # evaluate predictions days for each week
        predictions = np.array(predictions)
        score, scores = self.evaluate_forecasts(test[:, :, 0], predictions)

        return score, scores

    def build_model(self, train: np.ndarray, loss: tf.keras.losses.Loss,
                    activation='linear', neurons_per_layer: List[int] = None) \
            -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """ Train the model """

        if neurons_per_layer is None:
            neurons_per_layer = [200, 200, 100]

        # prepare data
        num_features = train.shape[2]
        train_x, train_y = self.to_supervised(train, self.window_size, self.window_size)
        # define parameters
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

        # define model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(neurons_per_layer[0], activation='relu',
                                       input_shape=(self.window_size, num_features)))
        model.add(tf.keras.layers.RepeatVector(n_outputs))
        model.add(tf.keras.layers.LSTM(neurons_per_layer[1], activation='relu',
                                       return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(neurons_per_layer[2],
                                                                        activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_features,
                                                                        activation=activation)))
        model.compile(loss=loss, optimizer='adam')

        return model, train_x, train_y

    @classmethod
    def train_model(cls, model: tf.keras.Model, train_x: np.ndarray, train_y: np.ndarray, epochs: int,
                    batch_size: int, verbose: int) -> tf.keras.Model:

        # Reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

        # Fit the Model
        model.fit(train_x, train_y, epochs=epochs,
                  batch_size=batch_size, verbose=verbose)
        return model

    @staticmethod
    def to_supervised(train: np.ndarray, n_input: int, n_out: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """ Convert history into inputs and outputs """

        # flatten data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        x, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(data):
                x.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1

        train_x, train_y = np.array(x), np.array(y)

        return train_x, train_y

    def generate_predictions_one_by_one(self, train: np.ndarray, test: np.ndarray,
                                        model: tf.keras.Model) -> np.ndarray:

        # history is a list of weekly data
        history = [x for x in train]
        # walk-forward validation over each week

        predictions = list()
        num_groups = test.shape[0]
        num_days_per_group = test.shape[1]
        num_features = test.shape[2]
        test_flat = test.reshape((test.shape[0] * test.shape[1], test.shape[2]))
        test2 = []
        for ind, _ in enumerate(test_flat[:]):
            test_group = []
            for test_days in test_flat[ind:ind + num_days_per_group]:
                test_group.append(test_days)
            if len(test_group) < num_days_per_group:
                test_group += [np.empty(num_features) for _ in
                               range(num_days_per_group - len(test_group))]
            test2.append(np.array(test_group))
        test2 = np.array(test2)

        for i in range(len(test2)):
            # predict the week
            yhat_sequence = self.forecast(model, history, self.window_size)
            # store the predictions
            predictions.append(yhat_sequence[0])
            extra_seq_diff = i - ((num_groups - 1) * num_days_per_group + 1)
            if extra_seq_diff > 0:
                test2[i, -extra_seq_diff:] = yhat_sequence[-extra_seq_diff:]
            # get real observation and add to history for predicting the next week
            history.append(test2[i, :])
        predictions = np.array(predictions)
        predictions = predictions.reshape((num_groups, predictions.shape[0] // num_groups,
                                           predictions.shape[1]))
        return predictions

    def generate_predictions_whole_group(self, train: np.ndarray, test: np.ndarray,
                                         model: tf.keras.Model) -> np.ndarray:

        # history is a list of weekly data
        history = [x for x in train]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(test)):
            # predict the week
            yhat_sequence = self.forecast(model, history, self.window_size)
            # store the predictions
            predictions.append(yhat_sequence)
            # get real observation and add to history for predicting the next week
            history.append(test[i, :])
        predictions = np.array(predictions)
        return predictions

    @staticmethod
    def forecast(model: tf.keras.Model, history: List[np.ndarray], n_input: int) \
            -> np.ndarray:
        """ Make a forecast """

        # flatten data
        data = np.array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-n_input:, :]
        # reshape into [1, n_input, n]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    @staticmethod
    def evaluate_forecasts(actual: np.ndarray, predicted: np.ndarray) \
            -> Tuple[float, List[float]]:
        """ evaluate one or more weekly forecasts against expected values """

        scores = list()
        # calculate an RMSE score for each day
        for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i, :], predicted[:, i, :])
            print(mse)
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)

        # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                for feat in range(actual.shape[2]):
                    s += (actual[row, col, feat] - predicted[row, col, feat]) ** 2
        score = sqrt(s / (actual.shape[0] * actual.shape[1] * actual.shape[2]))
        return score, scores
