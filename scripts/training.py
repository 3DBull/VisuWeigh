"""
VisuWeigh Project
This is the training script.
The script can be called as a function or as an API server.
Each model in the directory will be fit to the data.

Parameters:
    model_directory - directory in which the model files are stored. These model files should be .py files
        and return a valid keras model.




Derek Syme
2022-06-02
"""

### SETTINGS ####
SELECT_GPU = 0
HOST = 'localhost'
PORT = 6002
TOENSORBOARD_PORT = 6006
#################

import os
import logging
import tensorflow as tf
# fix some memory expansion issues by allowing growth on the gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info(gpus)
tf.config.experimental.set_visible_devices(gpus[SELECT_GPU], 'GPU')

from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import keras
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm
import keract
from matplotlib import pyplot as plt
from generator import Generator
from lib.util import test_on_image, load_and_prep_data
import json
import pandas as pd
from lib import paths 

# fix some memory expansion issues by allowing growth on the gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def load_data(path)
    '''
    
    '''
    with open(os.path.join(paths.TRAINING_DATA, paths.DATASET), 'r') as file:
        frame = json.load(file)

    df = pd.DataFrame(frame)
    df.timestamp = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# drop data with no image
def has_image(path):
    return os.path.exists(path)

dne = df[~df.path.apply(has_image)]
df = df[df.path.apply(has_image)]