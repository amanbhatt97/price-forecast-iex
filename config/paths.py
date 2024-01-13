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
        self.forecasts_dam = os.path.join(parent_directory, 'forecasts', 'day_ahead')  # DAM forecast files
        self.forecasts_dir = os.path.join(parent_directory, 'forecasts', 'directional')  # Directional forecast files
        self.reports_dam = os.path.join(parent_directory, 'reports', 'day_ahead')  # DAM reports
        self.reports_dir = os.path.join(parent_directory, 'reports', 'directional')  # Directional reports

        # Create directories if they do not exist
        self._create_directories()


    def _create_directories(self):
        """
        Create necessary directories if they do not exist.
        """
        directories = [
            self.data, self.src, self.deploy, self.config, self.logs, self.models,
            self.forecasts_dam, self.forecasts_dir, self.reports_dam, self.reports_dir
        ]

        # Create directories if they do not exist
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Create an instance of the ProjectPaths class
project_paths = ProjectPaths(PROJECT_PATH)

# Data path
raw_data_path = os.path.join(project_paths.data, 'raw')
processed_data_path = os.path.join(project_paths.data, 'processed')
external_data_path = os.path.join(project_paths.data, 'external')
