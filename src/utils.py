import pandas as pd
import os, logging
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
    Save the data as pickle.

    Args:
        file_name (str): The name of the pickle file.
        path (str): The path to save the pickle file. 

    Returns:
        pd.DataFrame: Save the DataFrame to pickle.
    """
    with open(os.path.join(path, f'{file_name}'), 'wb') as file:
        pickle.dump(data, file)

def save_excel(data, path, file_name):
    """
    Save the data as excel.

    Args:
        file_name (str): The name of the file.
        path (str): The path to save the excel file. 

    Returns:
        pd.DataFrame: Save the DataFrame to excel.
    """
    data.to_excel(os.path.join(path, f'{file_name}.xlsx'), index=False)

def load_excel(path, file_name):
    """
    Reads excel file and joins it using the specified path.

    Args:
        file_name (str): The name of the excel file.
        path (str): The path where the excel file is located. 

    Returns:
        pd.DataFrame: The DataFrame read from the excel file.
    """
    file_path = os.path.join(path, f'{file_name}.xlsx')
    return pd.read_excel(file_path)

def configure_logger(log_path, logger_type):
    logger = logging.getLogger(f'{logger_type}_logger')
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(log_path, f'{logger_type}'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger