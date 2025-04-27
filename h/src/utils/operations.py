def load_data(self):

    """

    Loads and processes data from a CSV file specified in the configuration.
    s
    This method reads a CSV file using a file path from the class configuration, processes it based on predefined rules, and stores the result in class attributes.

    :raises FileNotFoundError: If the CSV file cannot be found at the path provided in the configuration.
    :raises ColumnMismatchError: If the actual and expected column counts do not match.
    :raises KeyError: If a required configuration key is missing.

    """

    try:
        self.data = pd.read_csv(self.config['DATASET_PATH'], 
                                dtype=str, 
                                comment='#', 
                                float_precision='high', 
                                header=None).astype(np.float64)

        input_cols = self.config['INPUT_COLS']

        if self.config['CONFIGURATION'] == "I" :
            time_step = self.data.iloc[0, len(input_cols)]
            time_final = self.data.iloc[0, len(input_cols) + 1]
            num_time_steps = int(time_final/time_step) + 1
            self.time = np.arange(0, time_final, time_step)

            time_columns = [f"Time={1e3 * time_step * i:.2f}ms" for i in range(num_time_steps)]
            self.data.columns = input_cols + ['ts', 'tf'] + time_columns
        
        else:
            self.data.columns = input_cols + ['sensitivity']

        self.data.to_csv("C_nromalized.csv", index=False)

    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to locate the data file at '{self.config[file_name]}'. Verify file path in configuration.")

    except ColumnMismatchError:
        expected_cols = len(column_labels)
        found_cols = len(self.data.columns)
        raise ColumnMismatchError(f"Column count mismatch. Expected {expected_cols}, found {found_cols}. Check 'INPUT_COLS' in configuration.")
        

    import pandas as pd

data = pd.read_csv('./data/C_training_lhs.csv')

# Normalize the first three columns using standard normalization
columns_to_normalize = ['Overetch', 'Offset', 'Thickness']
for column in columns_to_normalize:
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std

# Keep 'ts' and 'tf' as they are
# Normalize the "Values" column(s) with respect to the maximum value across all rows
value_columns = [col for col in data.columns if (col.startswith('Time'))]
all_values = data[value_columns].values.flatten()
max_value = np.max(all_values)
min_value = np.min(all_values)

for column in value_columns:
    data[column] = (data[column] - min_value) / (max_value - min_value)

# Save the normalized dataset to a new CSV file
data.to_csv('normalized_dataset.csv', index=False)