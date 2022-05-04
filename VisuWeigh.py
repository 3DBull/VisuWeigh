'''
VisuWeigh version 1.0
'''
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from data_processing.yolo import Predictor
from cv2 import cv2
import json

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
print(f"Tensorflow={tf.version.VERSION}")

WEIGH_MODEL = 'best_model'
COW_DETECTION_THRESH = 0.10
MODELS_PATH = "C:/Users/XYZ/PycharmProjects/pythonProject/models"
XPAD = 10
YPAD = 20

VERSION = 1.0

# not implemented
MULTIVIEW = False

predictor = Predictor(MODELS_PATH + '/yolov3.weights', obj_thresh=COW_DETECTION_THRESH)
weigh_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, WEIGH_MODEL))


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

    # img = tf.keras.preprocessing.image.load_img(img_path, target_size=None, color_mode='rgb')
    # tf.image.pad_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    img = expand2square(img, 0)
    x = cv2.resize(img, target_size)
    # img = tf.image.resize(img, size=target_size, preserve_aspect_ratio=True, antialias=False)

    # x = tf.keras.preprocessing.image.img_to_array(img)

    im_num += 1
    # if predictor model has 3 color channels
    if weigh_model.input_shape[3] == 3:
        x = np.stack([x, x, x], axis=2)

    # print(f'stacked shape: {x.shape}')
    # x = np.expand_dims(x, axis=0)
    # print(x.shape)
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

def save(image, count, weight, p_weight):
    print('Saving...')
    cv2.imwrite(f'client_data/img/img_{count}.png', image)
    with open(f'../client_data/client_data.json', 'r+') as f:
        data = json.load(f)
        data.append({'img_id': count, 'weight': weight, 'predicted': str(p_weight)})
        jasonf = json.dumps(data)
        f.seek(0)
        f.write(jasonf)
    return

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
    # TODO add error handling for input
    if len(images) == 0:
        if len(im_paths) > 0:
            images = [cv2.imread(im_path) for im_path in im_paths]
        else:
            raise ValueError('Provide an input to the predictor!')

    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # predict the main bounding box and preprocess image
    '''
    "prediction": 
    [{"Label": "cow", "Prob": "0.7044643", "Box": {"x": [284, 103], "y": [340, 188]}}, 
    {"Label": "cow", "Prob": "0.10411254", "Box": {"x": [360, 111], "y": [410, 184]}},]
    '''
    predictions = [predictor.predict_cow_info(image=image) for image in images]
    #cv2.imshow('image', gray)
    #print(f'gray shape{gray.shape}')

    # TODO handle no-cow in image
    if predictions == [] or predictions == [[]]:
        print('Empty prediction!')
        return [], images, []

    # capture individual cows out of image
    crops = [[getcrop(gray, pred) for pred in prediction] for gray, prediction in zip(grays, predictions)]

    # concatenate images as batch
    X = [np.asarray([preprocess(im, target_size=weigh_model.input_shape[1:3]) for im in images]) for images in crops]
    #print(X.shape)
    if MULTIVIEW:
        # TODO
        # X = np.concatenate([X[0], X[1], np.zeros(X[0].shape)], axis=3)
        pass

    # predict the weight
    if X == []:
        print('No cows! Returning..')
        return [], images, []

    weights = [weigh_model.predict(_x) for _x in X]
    weights = [[w[0] for w in weight] for weight in weights] # repack weights

    # calculate confidence
    # TODO

    # render bounding box and weight on original image
    images = [drawboxes(image, prediction, weight) for image, prediction, weight in zip(images, predictions, weights)]

    return crops, images, weights,  # confidence
