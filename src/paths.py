""" 
This script defines the path for different directories/files present in this project. 
"""

#----- dependencies -----#

import os,sys   # working with file paths and the file system



#----- parent directory -----#

script_dir = os.path.dirname(os.path.realpath(__file__))    # current script's directory
project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))       # get the parent directory
sys.path.append(project_dir)     # add the parent directory to the system path



class ProjectPaths:

    def __init__(self, parent_directory):

        # store the parent directory
        self.parent_directory = parent_directory

        # define paths for various directories
        self.data = os.path.join(parent_directory, 'data')      # store data
        self.models = os.path.join(parent_directory, 'src')     # source scripts
        self.deploy = os.path.join(parent_directory, 'deploy')      # forecasting/reports scripts
        self.config = os.path.join(parent_directory, 'config')      # configuration files

        self.dam_logs = os.path.join(parent_directory, 'logs', 'day_ahead')     # logs for dam
        self.dir_logs = os.path.join(parent_directory, 'logs', 'directional')     # logs for directional

        self.forecasts_dam = os.path.join(parent_directory, 'forecasts', 'day_ahead')   # dam forecast files
        self.forecasts_dir = os.path.join(parent_directory, 'forecasts', 'directional')     # directional forecast files

        self.reports_dam = os.path.join(parent_directory, 'reports', 'day_ahead')       # dam reports
        self.reports_dir = os.path.join(parent_directory, 'reports', 'directional')     # directional reports

        # create directories if they do not exist
        self.create_directories()


    def create_directories(self):
        """
        create necessary directories if they do not exist.
        """
        directories = [
            self.data, self.models, self.deploy, self.config, self.dam_logs, self.dir_logs,
            self.forecasts_dam, self.forecasts_dir, self.reports_dam, self.reports_dir
        ]

        # create directories if they do not exist
        for directory in directories:
            os.makedirs(directory, exist_ok=True)



# create an instance of the ProjectPaths class
project_paths = ProjectPaths(project_dir)  



# forecast paths
dam_forecast_path = project_paths.forecasts_dam
dir_forecast_path = project_paths.forecasts_dir 


# accuracy reports path
dam_reports_path = project_paths.reports_dam
dir_reports_path = project_paths.reports_dir

