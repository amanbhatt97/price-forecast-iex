"""
Path definitions for various directories/files in the project.

Author: Aman Bhatt
"""

from pathlib import Path
import sys, os

from dotenv import load_dotenv
load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)


class ProjectPaths:
    def __init__(self, parent_directory):
        """
        Initialize ProjectPaths class.

        Args:
            parent_directory (Path): The parent directory of the project.
        """
        # Store the parent directory
        self.parent_directory = parent_directory

        # Define paths for various directories
        self.data = os.path.join(parent_directory, 'data')  # Store data
        self.src = os.path.join(parent_directory, 'src')  # Source scripts
        self.deploy = os.path.join(parent_directory, 'deploy')  # Forecasting/reports scripts
        self.config = os.path.join(parent_directory, 'config')  # Configuration files
        self.logs = os.path.join(parent_directory, 'logs')       # Logs
        self.models = os.path.join(parent_directory, 'models')   # saved models 
        self.forecasts = os.path.join(parent_directory, 'forecasts')  # forecast files
        self.reports = os.path.join(parent_directory, 'reports')  # reports

        # Create directories if they do not exist
        self._create_directories()


    def _create_directories(self):
        """
        Create necessary directories if they do not exist.
        """
        directories = [
            self.data, self.src, self.deploy, self.config, self.logs, self.models,
            self.forecasts, self.reports
        ]

        # Create directories if they do not exist
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# create an instance of the ProjectPaths class
project_paths = ProjectPaths(PROJECT_PATH)

# data path
RAW_DATA_PATH = os.path.join(project_paths.data, 'raw')
PROCESSED_DATA_PATH = os.path.join(project_paths.data, 'processed')
EXTERNAL_DATA_PATH = os.path.join(project_paths.data, 'external')

# model path
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

# dam forecast path
DAM_FORECAST_PATH = os.path.join(project_paths.forecasts, 'day_ahead')
DIR_FORECAST_PATH = os.path.join(project_paths.forecasts, 'directional') 