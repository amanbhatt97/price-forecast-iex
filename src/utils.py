import pandas as pd
import os
import pickle

def load_pickle(path, file_name):
    """
    Reads a pickle file and joins it using the specified path.

    Args:
        file_name (str): The name of the pickle file.
        path (str): The path where the pickle file is located. 

    Returns:
        pd.DataFrame: The DataFrame read from the pickle file.
    """
    file_path = os.path.join(path, f'{file_name}')
    return pd.read_pickle(file_path)

def save_pickle(data, path, file_name):
    """
    Reads a pickle file and joins it using the specified path.

    Args:
        file_name (str): The name of the pickle file.
        path (str): The path to save the pickle file. 

    Returns:
        pd.DataFrame: Save the DataFrame to pickle.
    """
    with open(os.path.join(path, f'{file_name}'), 'wb') as file:
        pickle.dump(data, file)