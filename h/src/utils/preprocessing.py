import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class preprocessing:
    
    """
    Class to handle data Processing.
    """

    
    def __init__(self, config_path):

        with open(config_path, 'r') as file:
            self.config = json.load(file)

        self.data = None
        self.time = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_vec = self.X_test_vec =  self.y_train_vec = self.y_test_vec = None
        self.process()

    def process(self):
        self.load_data()
        self.split_data()

    def load_data(self):
        self.data = pd.read_csv(self.config["DATASET_PATH"])

    def split_data(self):

        """

        Splits the dataset into training and test sets based on the configuration.

        This method uses class attributes for the dataset and configuration. It assumes that 'INPUT_COLS' and 'TEST_SIZE' are defined in the configuration.
        
        """

        if len(self.data.columns) > 5:
            output_cols = self.data.columns[5:-1]
            self.time = np.arange(0.,1.5,1e-2)
        else:
            output_cols = self.data.columns[-1]

        X = self.data[self.config['INPUT_COLS']].values
        y = self.data[output_cols].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config['TEST_SIZE'], random_state=self.config['RANDOM_STATE']
        )

        self.X_train, self.y_train = self.shuffle_data(self.X_train, self.y_train)

        if self.time is not None:
            X_train_rep, y_train_rep = self.stack_data(self.X_train, self.y_train, self.time)
            X_test_rep, y_test_rep = self.stack_data(self.X_test, self.y_test, self.time)
            self.X_train_vec = X_train_rep
            self.X_test_vec = X_test_rep
            self.y_train_vec = y_train_rep
            self.y_test_vec = y_test_rep
        else:
            self.X_train_vec = self.X_train
            self.X_test_vec = self.X_test
            self.y_train_vec = self.y_train
            self.y_test_vec = self.y_test

    @staticmethod
    def stack_data(X, y, time):

        """

        Prepares time series data by expanding the input features (X) and flattening the output features (y).
        
        This method repeats each row of X for the number of time steps, and appends the corresponding time step to each repeated row. The output array y is flattened.

        :param X: (numpy array) the input features with shape (n_samples, n_features).
        :param y: (numpy array) the output features with shape (n_samples, n_time_steps).
        :param time: (numpy array) the time steps with length equal to n_time_steps.

        :return X_rep: (numpy array) expanded X with time steps appended.
        :return y_rep: (numpy array) flattened y.

        """

        output_cols = y.shape[1]
        X_rep = np.repeat(X, output_cols, axis=0)
        time_repeated = np.tile(time, len(X_rep) // len(time))
        X_rep = np.column_stack((X_rep, time_repeated))
        y_rep = y.flatten()
        return X_rep, y_rep


    @staticmethod
    def shuffle_data(X, y):

        """

        Randomly shuffles the data.

        :param X: Input features.
        :param y: Output labels.

        :return: Shuffled input features and corresponding labels.

        """

        perm = np.random.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        return X_shuffled, y_shuffled
    
class LSTMDataProcessor(preprocessing):
    def __init__(self, config_path):
        super().__init__(config_path)

    @staticmethod
    def VoltageProfile(t, amplitude_val=1.8, Tx_val=0.4e-3):
        return 1.8 * amplitude_val * (1 + np.sin((t / Tx_val - 1 / 4) * 2 * np.pi)) * (t < 2 * Tx_val)

    @staticmethod
    def stack_data(X, y, time):
        """
        Reshapes data into sequences for LSTM training, including the VoltageProfile feature.
        Parameters:
        - X: The input features, expected shape [n_samples, n_features].
        - y: The output features, expected shape [n_samples, output_features].
        - time_steps: The number of time steps to be used in each sequence.
        Returns:
        - Two numpy arrays: X_seq and y_seq, reshaped for LSTM.
        """
        n_samples, n_features = X.shape
        X_seq = np.zeros((n_samples, len(time), n_features + 2))
        y_seq = np.zeros((n_samples, len(time), 1))

        for i in range(n_samples):
            for t in range(len(time)):
                X_seq[i, t, :-2] = X[i]
                X_seq[i, t, -2] = time[t]
                X_seq[i, t, -1] = LSTMDataProcessor.VoltageProfile(time[t])  # Assuming each time step is 1e-5
                y_seq[i, t, :] = y[i,t]
        return X_seq, y_seq


    def scale_data(self, scaling_strategy=None):
        """
        Scales 3D LSTM data and then reshapes it back to 3D.
        Parameters:
        - X: 3D data array of shape [n_samples, n_timesteps, n_features]
        - scaler: An instance of a scaler, e.g., StandardScaler or MinMaxScaler
        - n_timesteps: Number of timesteps in each sequence
        Returns:
        - Scaled and reshaped data
        """

        self.X_train, self.y_train = self.shuffle_data(self.X_train, self.y_train)
        X_train_rep, y_train_rep = self.stack_data(self.X_train, self.y_train, self.time)
        X_test_rep, y_test_rep = self.stack_data(self.X_test, self.y_test, self.time)

        n_samples, n_timeSteps, n_features = X_train_rep.shape
        n_samples_test = X_test_rep.shape[0]

        # Flatten to 2D
        X_train_rep = X_train_rep.reshape(-1, n_features)
        X_test_rep = X_test_rep.reshape(-1, n_features)
        y_train_rep = y_train_rep
        y_test_rep = y_test_rep

        if scaling_strategy is None or scaling_strategy == 'standard':
            self.scaler = StandardScaler()
        elif scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()

        self.X_train_scaled = self.scaler.fit_transform(X_train_rep).reshape(n_samples, n_timeSteps, n_features)
        self.X_test_scaled = self.scaler.transform(X_test_rep).reshape(n_samples_test, n_timeSteps, n_features)
        self.y_train_scaled =  y_train_rep
        self.y_test_scaled =  y_test_rep
