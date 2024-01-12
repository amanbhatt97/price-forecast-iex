# Price Forecast IEX

## Overview

The **Price Forecast IEX** project is designed to forecast energy prices, utilizing data from the IEX (Indian Energy Exchange), weather data and power satations data from different locations. The project focuses on day-ahead and real-time market forecasts, employing machine learning techniques for accurate predictions.

## Table of Contents

- [Introduction](#price-forecast-iex)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Training Models](#training-models)
- [Forecasting](#forecasting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

- Python 3
- Required Python packages (specified in requirements.txt)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/price-forecast-iex.git
   cd price-forecast-iex

2. Create a Virtual Environment (Optional but Recommended):

   ```bash
   python3 -m venv env

3. Activate the virtual environment:

   ```bash
   # On Windows
   .\venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate  

4. Install Dependencies:

   ```bash
   pip install -r requirements.txt

## Usage

The project is structured to facilitate both the training of models and the forecasting process. Detailed instructions for training and forecasting can be found in the respective sections below.

## Project Structure

The repository is organized into distinct modules to handle data processing, feature engineering, model training, and forecasting. 

* [config/](./config)
  * [locations.yaml](./config/locations.yaml):  
  * [paths.py](./config/paths.py)
  * [utils.py](./config/utils.py)
* [data/](./data)
  * [external/](./data/external)
  * [processed/](./data/processed)
  * [raw/](./data/raw)
* [deploy/](./deploy)
  * [dam_train.py](./deploy/dam_train.py)
* [env/](./env)
* [forecasts/](./forecasts)
* [logs/](./logs)
* [models/](./models)
* [notebooks/](./notebooks)
  * [dam_train.ipynb](./notebooks/dam_train.ipynb)
* [reports/](./reports)
* [src/](./src)
  * [data_ingestion/](./src/data_ingestion)
    * [iex_data.py](./src/data_ingestion/iex_data.py)
    * [weather_data.py](./src/data_ingestion/weather_data.py)
  * [feature_engineering/](./src/feature_engineering)
  * [model_building/](./src/model_building)
* [.env](./.env)
* [.gitignore](./.gitignore)
* [LICENSE](./LICENSE)
* [README.md](./README.md)

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We acknowledge the use of data from the Indian Energy Exchange (IEX) and appreciate the efforts of contributors to open-source libraries and tools used in this project.
