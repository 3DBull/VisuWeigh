'''
The VisuWeigh Project

This module is used to weigh a cow from on image.

Updated 2022-06-20
By Derek Syme
'''
import logging
import os
import tensorflow as tf
import numpy as np
from VisuWeigh.lib.yolo import Predictor
from cv2 import cv2
import json
from datetime import datetime as dt
from VisuWeigh.lib import paths

LOGGER = logging.getLogger(__name__)

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


LOGGER.info(f"Tensorflow={tf.version.VERSION}")

### FILE SETTINGS ###
COW_DETECTION_THRESH = 0.10
XPAD = 10
YPAD = 20
MULTIVIEW = False # Not implemented
VERSION = 1.0
####################

predictor = Predictor(paths.YOLO, obj_thresh=COW_DETECTION_THRESH)
weigh_model = tf.keras.models.load_model(paths.SERV_MODEL)


def getcrop(img, pred):
    global XPAD, YPAD

    box = pred['Box']
    img = img[max(pred['Box']['x'][1] - YPAD * 2, 0): min(pred['Box']['y'][1] + YPAD, img.shape[0]),
          max(pred['Box']['x'][0] - XPAD * 4, 0): min(pred['Box']['y'][0] + XPAD, img.shape[1])]

    return img


# Add padding to image
# Function adapted from:
# https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
def expand2square(img, background_color):
    width, height = img.shape[:2]
    if width == height:
        return img
    elif width > height:
        result = np.zeros((width, width), dtype='uint8')
        pad = (width - height) // 2
        r_pad = width - height - pad
        result[:, pad: width - r_pad] = img
        return result
    else:
        result = np.zeros((height, height), dtype='uint8')
        pad = (height - width) // 2
        b_pad = height - width - pad
        result[pad:height - b_pad, :] = img
        return result


im_num = 0


def preprocess(img, target_size=(224, 224)):
    global im_num
    # TODO # Add error handling for input

    img = expand2square(img, 0)
    x = cv2.resize(img, target_size)

    im_num += 1

    # if predictor model has 3 color channels
    if weigh_model.input_shape[3] == 3:
        x = np.stack([x, x, x], axis=2)

    # LOGGER.debug(f'stacked shape: {x.shape}')
    # x = np.expand_dims(x, axis=0)
    x = x / 255.
    return x


def drawboxes(img, pred, weight):
    for p, w in zip(pred, weight):
        x1, y1 = p['Box']['x']
        x2, y2 = p['Box']['y']

        textsize = (img.shape[0]+img.shape[1])/1000.
        cv2.rectangle(img, (x1, y1 - int(20*textsize)), (x2, y2), (174, 149, 105), int(textsize*3))
        cv2.putText(img, 'Weight: {:.1f}'.format(w), (x1 + 11, y1 - 35 - int(textsize/4)), 0, textsize/2, (0, 0, 0), int(textsize*1.2))
        cv2.putText(img, 'Weight: {:.1f}'.format(w), (x1 + 10, y1 - 36 - int(textsize/4)), 0, textsize/2, (174, 149, 105), int(textsize))

    return img


def init_client_dir(path):
    if not os.path.exists(path):
        LOGGER.warning(f'Could not find client data folder at {path}. Creating a new directory.')
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'img'))
        with open(os.path.join(path, 'pcount.txt'), 'w') as f:
            f.write(0)
        with open(os.path.join(path, 'client_data.json'), 'w') as f:
            f.write('[]')


def save_client_info(image, save_dir, count, weight, p_weight):
    LOGGER.info('Saving client data ... ')
    try:
        cv2.imwrite(os.path.join(save_dir, 'img', f'im_{count}.png'), image)
        with open(os.path.join(save_dir, 'client_data.json'), 'r+') as f:
            data = json.load(f)
            data.append({'time': dt.now().strftime('%Y-%m-%d-%H:%M:%S.%f'), 'impath': os.path.join(f'im_{count}.png'), 'weight': weight, 'predicted': str(p_weight)})
            jasonf = json.dumps(data)
            f.seek(0)
            f.write(jasonf)
    except cv2.error as er:
        LOGGER.error(f'Could not save data. {er}')
        return False
    return True


def predict(im_paths=[], images=[], url_path=None):
    global weigh_model, predictor
    '''
    This is the primary function that will return an image with the bounding box and predicted weight rendered on the image.
    It will also return the predicted weight as well as the confidence of the prediction.
    :param image: The input image of a cow to predict the weight of
    :return: (rendered_image, prediction, confidence)
    '''
    # TODO add functionality for processing multiple images at a time

    # load the image from file (if necessary)
    if len(images) == 0:
        if len(im_paths) > 0:
            images = [cv2.imread(im_path) for im_path in im_paths]
        else:
            raise ValueError('Provide an input to the predictor!')
    try:
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        # predict the main bounding box from YOLO and preprocess image
        '''
        "prediction": 
        [{"Label": "cow", "Prob": "0.7044643", "Box": {"x": [284, 103], "y": [340, 188]}}, 
        {"Label": "cow", "Prob": "0.10411254", "Box": {"x": [360, 111], "y": [410, 184]}},]
        '''

        predictions = [predictor.predict_cow_info(image=image) for image in images]
        #cv2.imshow('image', gray)
        #LOGGER.debug(f'gray shape{gray.shape}')

        # TODO handle no-cow in image
        if predictions == [] or predictions == [[]]:
            LOGGER.error('Empty prediction!')
            return [], images, []

        # capture individual cows out of image
        crops = [[getcrop(gray, pred) for pred in prediction] for gray, prediction in zip(grays, predictions)]

        # concatenate images as batch
        X = [np.asarray([preprocess(im, target_size=weigh_model.input_shape[1:3]) for im in images]) for images in crops]

        #LOGGER.debug(X.shape)
        if MULTIVIEW:
            # TODO
            # X = np.concatenate([X[0], X[1], np.zeros(X[0].shape)], axis=3)
            pass

        # predict the weight
        if X == []:
            LOGGER.error('No cows! Returning..')
            return [], images, []

        weights = [weigh_model.predict(_x) for _x in X]
        weights = [[w[0] for w in weight] for weight in weights] # repack weights

        # calculate confidence
        # TODO

        # render bounding box and weight on original image
        images = [drawboxes(image, prediction, weight) for image, prediction, weight in zip(images, predictions, weights)]
    except Exception as ex:
        LOGGER.exception(f'Could not complete prediction: {ex}')
        return [], [], []
    return crops, images, weights,  # confidence
