import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from typing import List, Optional, Callable, Tuple
import numpy as np
from tensorflow.keras import losses

# Retrieve the loss function object based on the string identifier
from scipy.optimize import minimize
from model import NN_Model

class DeepONet(NN_Model):
    def __init__(self, trunk_config: dict, branch_config: dict, output_units: int = 1):
        super().__init__()
        self.trunk_config = trunk_config
        self.branch_config = branch_config
        self.output_units = output_units
        self.trunk_net = NN_Model()
        self.branch_net = NN_Model()

    def build_model(self):
        """
        Build the DeepONet by combining trunk and branch networks.
        """
        # Build trunk and branch networks with explicit input layers
        trunk_input = tf.keras.layers.Input(shape=(self.trunk_config["input_shape"],), name="Trunk_Input")
        self.trunk_net.build_model(**self.trunk_config)
        trunk_output = self.trunk_net.model(trunk_input)

        branch_input = tf.keras.layers.Input(shape=(self.branch_config["input_shape"],), name="Branch_Input")
        self.branch_net.build_model(**self.branch_config)
        branch_output = self.branch_net.model(branch_input)

        # Combine trunk and branch outputs
        combined_output = tf.keras.layers.Dot(axes=1)([trunk_output, branch_output])   
             
        # Final output layer
        final_output = tf.keras.layers.Dense(self.output_units, activation="linear", name="Final_Output")(combined_output)

        # Create final model
        self.model = tf.keras.models.Model(inputs=[trunk_input, branch_input], outputs=final_output)

    def train_model(self, trunk_data, branch_data, y, trunk_val, branch_val, y_val, **kwargs):
        """
        Train the DeepONet using trunk and branch data.

        :param trunk_data: Input data for the trunk net.
        :param branch_data: Input data for the branch net.
        :param y: Target data.
        :param trunk_val: Validation data for the trunk net.
        :param branch_val: Validation data for the branch net.
        :param y_val: Validation target data
        """
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])
        self.history = self.model.fit(
            [trunk_data, branch_data], y,
            validation_data=([trunk_val, branch_val], y_val),
            **kwargs
        )

    def save_model(self, model_path: str):
        """
        Save the DeepONet model.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def predict(self, trunk_data, branch_data):
        """
        Make predictions using the DeepONet.

        :param trunk_data: Input data for the trunk net.
        :param branch_data: Input data for the branch net.
        :return: Predictions.
        """
        return self.model.predict([trunk_data, branch_data], verbose=0)