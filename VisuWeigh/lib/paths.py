'''
This file contains the path information for the VisuWeigh project

The data base directory for the project is structured as follows:

    root/
        cattle_data/
            raw/
                {auction}_{date}_{extension}.json
                ...

                img/
                    {auction}/
                        im_{imgae#}.png
                        ...

            training/
                img/
                    {singles}
                training.json

            evaluation/
                img/
                    im_{image#}.png
                easy/
                    easy_eval_{date}.json
                    ...
                hard/
                    hard_eval_{date}.json
                    ...
                results.csv


            models/
                input/
                    {model_name}.py
                    ...
                output/
                    {model_name}_{epoch}_{val_loss} #keras model
                    ...
                training_logs/
                    {model_name}_{parameter_name}_{date}/
                        train/
                        validation/
                        parameters.json
                    ...


Derek Syme
2022-06-03
'''

import os
import requests
import zipfile
import logging
import json
from VisuWeigh.lib import paths


def find_config(name):
    # TODO - Implement search for config so scripts and modules can be run from any level
    pass


CONFIG_PATH = '../config.json'

with open(CONFIG_PATH, "r") as jsonfile:
    config = json.load(jsonfile)

ROOT = config['Database']['root']
DATABASE_LOCATION = os.path.join(ROOT, 'cattle_data')

# RAW DATA
RAW_DATA = os.path.join(DATABASE_LOCATION, 'raw')
RAW_IMG = os.path.join(RAW_DATA, 'img')

# TRAINING
TRAINING_DATA = os.path.join(DATABASE_LOCATION, 'training')
TRAINING_IMAGES = os.path.join(TRAINING_DATA, 'img')
TRAINING_LOGS = os.path.join(DATABASE_LOCATION, 'models', 'training_logs')
DATASET = 'training.json'

# EVALUATION
EVALUATION_DATA = os.path.join(DATABASE_LOCATION, 'evaluation', 'datasets')
EVALUATION_IMG = os.path.join(DATABASE_LOCATION, 'evaluation', 'img')
EVALUATION_RESULTS = os.path.join(DATABASE_LOCATION, 'evaluation', 'results.csv')

# MODELS
MODEL_BUILDS = os.path.join(DATABASE_LOCATION, 'models', 'build')
TRAINED_MODELS = os.path.join(DATABASE_LOCATION, 'models', 'trained')
YOLO = os.path.join(DATABASE_LOCATION, 'models', 'yolov3.weights')

# The user can use the project config to overwrite the default path to the server model
path = config['Paths']['SERV_MODEL']
if path == "":
    SERV_MODEL = os.path.join(DATABASE_LOCATION, 'models', 'serv')
    SERV_MODEL = os.path.join(SERV_MODEL, os.listdir(SERV_MODEL)[0])
else:
    SERV_MODEL = path


def build_db():

    root = paths.ROOT
    '''
    Downloads VisuWeigh data and sets up a database directory for a VisuWeigh project
    
    Params:
    root: where to set up the data base
    '''
    try:
        url = 'https://VisuWay.tech/VisuWeigh/database.zip'
        data = requests.get(url)
        with zipfile.ZipFile(data) as z:
            z.extractall(root)
    except Exception as e:
        logging.log(logging.ERROR, 'Could not setup directory at the given root!')
