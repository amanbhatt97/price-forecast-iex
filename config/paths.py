"""
Path definitions for various directories/files in the project.

Author: Aman Bhatt
"""

from pathlib import Path
import sys

# Parent directory
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.append(project_dir)  # Add the parent directory to the system path


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
        self.data = parent_directory / 'data'  # Store data
        self.src = parent_directory / 'src'  # Source scripts
        self.deploy = parent_directory / 'deploy'  # Forecasting/reports scripts
        self.config = parent_directory / 'config'  # Configuration files
        self.logs = parent_directory / 'logs'  # Logs
        self.forecasts_dam = parent_directory / 'forecasts' / 'day_ahead'  # DAM forecast files
        self.forecasts_dir = parent_directory / 'forecasts' / 'directional'  # Directional forecast files
        self.reports_dam = parent_directory / 'reports' / 'day_ahead'  # DAM reports
        self.reports_dir = parent_directory / 'reports' / 'directional'  # Directional reports

        # Create directories if they do not exist
        self.create_directories()

    def create_directories(self):
        """
        Create necessary directories if they do not exist.
        """
        directories = [
            self.data, self.src, self.deploy, self.config, self.logs,
            self.forecasts_dam, self.forecasts_dir, self.reports_dam, self.reports_dir
        ]

        # Create directories if they do not exist
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Create an instance of the ProjectPaths class
project_paths = ProjectPaths(project_dir)

# Data path
data_path = project_paths.data

# Logs path
log_path = project_paths.logs

# Forecast paths
dam_forecast_path = project_paths.forecasts_dam
dir_forecast_path = project_paths.forecasts_dir

# Accuracy reports path
dam_reports_path = project_paths.reports_dam
dir_reports_path = project_paths.reports_dir