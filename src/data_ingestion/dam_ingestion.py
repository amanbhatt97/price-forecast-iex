"""
This script fetches day-ahead market (DAM) data, save raw data into data/raw,
process the data and save it into data/processed.

Author: Aman Bhatt
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials
base_url = os.getenv('base_url')
user_email = os.getenv('email')
user_password = os.getenv('password')

# Parent directory
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

# Custom modules
from config.paths import *

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(log_path, 'dam.log')  # Save logs to a file named 'error.log'
)


# Datetime variables
dam_historical = pd.read_pickle(os.path.join(data_path, 'processed', 'dam_data'))
last_date = dam_historical['datetime'].iloc[-1].strftime('%d-%m-%Y')

# Fetch data from last available data upto latest data available
start_date = dam_historical['datetime'].iloc[-1] + timedelta(days=1)
end_date = datetime.now() + timedelta(days=1)

# Dates in string format
start_date_str = start_date.strftime("%d-%m-%Y")
end_date_str = end_date.strftime("%d-%m-%Y")


# DAM data fetching API
def get_token(base_url):
    """
    Fetches the access token for DAM data API.

    Args:
        base_url (str): The base URL for the API.

    Returns:
        str: Access token for authentication.
    """
    try:
        url = base_url + 'login'
        email = user_email
        password = user_password
        data = {'email': email, 'password': password}
        req = requests.post(url=url, data=data, verify=True).json()
        return req['access_token']
    except Exception as e:
        logging.info("Error occurred while retrieving the access token:", str(e))

def get_dam_api(start_date_str, end_date_str, token):
    """
    Makes a GET request to the DAM API endpoint for fetching day-ahead market data.

    Parameters:
        start_date_str (str): Start date for data retrieval in string format (format: "dd-mm-yyyy").
        end_date_str (str): End date for data retrieval in string format (format: "dd-mm-yyyy").
        token (str): Access token for API authentication.

    Returns:
        dict: JSON response containing DAM data.
    """
    try:
        url = base_url + 'getMarketVolume'
        headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
        params = {"start_date": start_date_str, "end_date": end_date_str}
        r = requests.get(url=url, headers=headers, params=params)
        return r.json()
    except Exception as e:
        logging.info("Error occurred while retrieving data:", str(e))

def get_dam_actual():
    """
    Fetches DAM (Day-Ahead Market) actual data and logs information.

    Returns:
        pd.DataFrame: DataFrame containing the fetched DAM data.
    """
    token = get_token(base_url)
    data_dict = get_dam_api(start_date_str, end_date_str, token)
    raw_data = pd.DataFrame(data_dict['data'])
    raw_data.to_pickle(os.path.join(data_path, 'raw', 'dam'))
    return raw_data



