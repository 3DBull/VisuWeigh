

DATABASE_LOCATION = 'E:/cattle_data'
INPUT_PATH = DATABASE_LOCATION+'/img'
OUTPUT_PATH = DATABASE_LOCATION+'/training'
DATASET = 'cattle_data.json' #data will be read from this file if COMBINE_DATA_FILES is False
COMBINE_DATA_FILES = True
USE_GPU = False


import tensorflow as tf
import pandas as pd
import json
from yolo import Predictor
from time import time
import datetime as dt
import cv2
import numpy as np
from multiprocess import  Pool
from matplotlib import pyplot as plt
import PIL.Image
from IPython.display import display
from io import BytesIO
import datetime
import ipywidgets as widgets
from ipywidgets import *
import warnings
from tqdm.notebook import tqdm, trange


