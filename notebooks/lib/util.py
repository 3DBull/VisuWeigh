"""
UTILITIES PACKAGE

Contains helper functions for the VisuWeigh project


Derek Syme
2022-05-12
"""


import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm
import keract
from matplotlib import pyplot as plt
from generator import Generator
import json
import pandas as pd
from matplotlib.patches import Patch
from lib import paths


def display_activation(model, image_frame):
    """
    Displays heatmaps of each layer overlaid onto the original image.
    :param model: Keras compiled model
    :param image_frame: The pandas frame from which to get the X and y data to feed the model.
    """

    img_path = image_frame.path
    if os.path.exists(img_path):
        img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.

        activations = keract.get_activations(model, x, layer_names=None, nodes_to_evaluate=None, output_format='simple',
                                             nested=False, auto_compile=True)
        # keract.display_activations(activations, cmap=None, save=False, directory='.', data_format='channels_last', fig_size=(24, 24), reshape_1d_layers=False)
        keract.display_heatmaps(activations, x, save=False)
        # return heatmap, output


# Add padding to image
# Function taken from:
# https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
def expand2square(pil_img):
    background_color = (0)
    width, height = pil_img.size[:2]
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_img(img_path, target_size, color_mode):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=None, color_mode=color_mode)
    # img = expand2square(img)
    img = tf.image.resize(img, size=target_size, preserve_aspect_ratio=True, antialias=False)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.
    return x


def test_on_image(models, image_frame, model_labels=[], display_each=True, display_width=3):
    """
    Run model on set of images in the given pandas frame. The image frame must include a ['path'] column to the image as well as a
    ['weight'] column with the true values for the image.

    :param model: Keras compiled model
    :param image_frame: The pandas frame from which to get the X and y data to feed through the model.
    :param display_each: (optional) If this is True, each image in the frame will be displayed inline along with its given truth
    and prediction.
    :param display_activation: (optional) If this is True, an activation map will be overlayed onto the image to show
    "where the network is looking" for the given image. In this case, 'display_each' should also be set to True.
    :return: Prints the average error and accuracy of the given test set. Returns the accuracy score.
    """

    # handle model_labels length mismatch
    if len(model_labels) == 0:
        for i in range(len(models)):
            model_labels.append('Model ' + str(i))

    elif len(model_labels) != len(models):
        raise ValueError("Number of model labels does not match number of models")

    # display a loading bar
    if not display_each:
        pbar = tqdm(total=len(image_frame))

    else:
        display_width = min(display_width, len(image_frame))
        fig, ax = plt.subplots(max(len(image_frame) // display_width, 1), min(display_width, len(image_frame)),
                               figsize=(4.5 * display_width, 4 * len(image_frame) // display_width))
        fig.tight_layout()
        # plt.rcParams["figure.figsize"] = [12.2, 5.5]
        plt.rcParams["figure.autolayout"] = True

    error = np.zeros((len(models), len(image_frame)))
    pred = np.zeros(len(models))
    accum_err = np.zeros(len(models))
    non_existent = 0

    # test on all the models and all the images
    for i in range(len(image_frame)):

        img_path = image_frame.iloc[i].path

        if os.path.exists(img_path):
            x = load_img(img_path, target_size=(224, 224), color_mode='rgb')

            # get predictions from each model to be tested
            for j, model in enumerate(models):

                # set the color mode based on the number of color channels in the model
                if model.input_shape[3] == 1:
                    x_ = x[:, :, :, 1]
                else:
                    x_ = x

                pred[j] = model.predict(x_)[0][0]

            # display the results
            if display_each:
                barwidth = 16
                font_size = 12

                if type(ax) == np.ndarray:
                    if len(ax.shape) == 1:
                        a = ax[i]
                    else:
                        a = ax[i // 3][i % 3]
                else:
                    a = ax

                ax_ = a.twinx()
                ax_1 = ax_.twinx()
                im = a.imshow(np.squeeze(x_), extent=[0, 224, 0, 224])
                horiz = np.array(range(barwidth, len(pred) * barwidth + barwidth, barwidth))
                MAA = (1 - np.abs(np.full(len(pred), image_frame.iloc[i].weight) - np.array(pred)) / np.full(len(pred),
                                                                                                             image_frame.iloc[
                                                                                                                 i].weight)) * 100

                ax_.bar(x=horiz, tick_label=model_labels, width=barwidth, linewidth=2, edgecolor='black', height=MAA,
                        alpha=0.5)
                ax_1.bar(x=224 - np.flip(horiz, axis=0),
                         tick_label=[label + '_' for label in np.flip(model_labels, axis=0)], width=barwidth,
                         linewidth=2, color='orange', edgecolor='black', height=pred, alpha=0.5)
                ax_1.bar(x=224 - barwidth * (len(pred) + 1), width=barwidth * (len(pred) + 1), linewidth=2,
                         align='edge', color='red', edgecolor='white', height=image_frame.iloc[i].weight, alpha=0.2)

                # print accuracy values
                for k, val in enumerate(MAA):
                    ax_.text(horiz[k] - 4, val / 2 - 10, '{:.1f}%'.format(val), color='white', fontweight='bold',
                             rotation='vertical', size=font_size)

                # print weight values
                for k, val in enumerate(pred):
                    ax_1.text(224 - horiz[k] - 4, val / 2 - 200, '{:.1f}lb'.format(val), color='white',
                              fontweight='bold', rotation='vertical', size=font_size)

                # ax_.set_aspect('equal', adjustable=None, anchor=None, share=False)
                legend_elements = [Patch(facecolor='red', edgecolor='black',
                                         label='True Weight'),
                                   Patch(facecolor='orange', edgecolor='black',
                                         label='Predicted Weight'),
                                   Patch(facecolor='C0', edgecolor='black',
                                         label='Accuracy')]

                # ax_.set_ylabel('Accuracy')
                ax_.set_xticks(np.concatenate([horiz, 224 - np.flip(horiz, axis=0)], axis=0),
                               labels=np.concatenate([model_labels, model_labels], axis=0))
                ax_1.set_ylabel('lbs')
                ax_1.set_ylim(0, 1500)
                ax_.set_ylim(0, 120)
                ax_.set_yticks(range(0, 125, 25))
                a.set_yticks([])
                a.tick_params(axis='x', labelrotation=70)
                a.set_title('Actual Weight: {} lbs'.format(image_frame.iloc[i].weight))
                # print('Predicted: {} lbs, Actual: {} lbs'.format(pred, image_frame.iloc[i].weight))

            # accumulate the error over the set of inputs
            error[:, i] = image_frame.iloc[i].weight - pred
            accum_err += abs(image_frame.iloc[i].weight - pred) / image_frame.iloc[i].weight

        else:
            non_existent += 1

        if not display_each:
            pbar.update(1)

    if display_each:
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0., 0.92, 0.02, 0.102),
                   ncol=1, borderaxespad=0.)
        fig.suptitle('Model Comparison', fontsize=30, verticalalignment='top')
        # plt.xticks(rotation='vertical')
        plt.show()

    # calculate the error and accuracy as an average for the set
    # print('{} average Error Rate: {:.2f}%'.format(model_labels[j], error[j]*100 / (len(image_frame) - non_existent)))
    MAA = (1 - accum_err / (len(image_frame) - non_existent)) * 100

    stats = []
    for i, label in enumerate(model_labels):
        print('{} average accuracy: {:.2f}%'.format(label, MAA[i]))

        p = error[i, :]
        stats.append({'name': label, 'mean_abs_accuracy': MAA[i], 'mean_abs_error': np.mean(np.abs(p)),
                      'error_mean': np.mean(p), 'error_std': np.std(p), 'error_min': min(p), 'error_max': max(p)})

    return stats


# drop data with no image
def has_image(path):
    return os.path.exists(path)


def load_and_prep_data(path_to_database, dataset):
    '''
    Use this function to load training data. The data is loaded from the directory and file provided. 
    Data is only kept if it has a valid image and in the range 400-1400 lb. 
    
    :param path_to_database: 
    :param dataset: Json file containing records with column names 'weight', 'timestamp', and 'path' 
    :return: Pandas DataFrame
    '''
    # load training data
    with open(os.path.join(path_to_database, dataset), 'r') as file:
        frame = json.load(file)
    df = pd.DataFrame(frame)
    df.timestamp = pd.to_datetime(df['timestamp'], unit='ms')
    print(f'Found {len(df)} raw data points!')

    df.path = df.path.str.replace('singles', 'img')
    df = df[df.path.apply(has_image)]
    # trim the data to a constrained weight range
    df_trim = df[(df.weight > 400) & (df.weight < 1450)]
    print(f'Loaded {len(df_trim)} filtered data points!')

    return df_trim


def evaluate_model(model_path=None, model=None, df=None):
    '''
    Provide either a model or model_path to evaluate a model. If no dataframe is provided, 
    the default evaluation set will be used. 
    :param model: 
    :param model_path: 
    :param df: dataframe to evaluate on
    :return: results with the schema {}
    '''
    if model is None:
        if model_path is None:
            raise ValueError('Please Specify a model path or provide a model to the function!')
        else:
            model = tf.keras.models.load_model(model_path)

    if df is None:
        #df = pd.concat([load_all_json(os.path.join(paths.EVALUATION_DATA, 'easy')), load_all_json(os.path.join(paths.EVALUATION_DATA, 'hard'))])
        df = load_all_json(os.path.join(paths.EVALUATION_DATA, 'easy'))
        #with open(os.path.join(paths.EVALUATION_DATA)) as file:
        #    frame = json.load(file)
        #df = pd.DataFrame(frame)
        #df.timestamp = pd.to_datetime(df['timestamp'], unit='ms')
        #df = df[df.path.apply(has_image)]
        #df = df[(df.weight > 400) & (df.weight < 1450)]
        print(f'Found {len(df)} data points!')

    results = test_on_image([model], df, model_labels=[model_path.split('/')[-1]], display_each=False)
    del model

    return results


def evaluate_ensamble(models):
    with open(os.path.join(paths.DATABASE_LOCATION, paths.EVALUATION_DATA)) as file:
        frame = json.load(file)
    df = pd.DataFrame(frame)
    df.timestamp = pd.to_datetime(df['timestamp'], unit='ms')

    df = df[df.path.apply(has_image)]
    df = df[(df.weight > 400) & (df.weight < 1450)]
    print(f'Found {len(df)} data points!')

    results = test_on_image(models, df, display_each=False)

    return np.mean(results)


def load_all_json(data_path):

    f_names = []
    for name in os.listdir(data_path):
        if name.endswith('.json'):
            f_names.append(name)
    print('Found {} files.'.format(len(f_names)))

    # load the dataset
    with open(os.path.join(data_path, f_names.pop()), 'r') as file:
        frame = json.load(file)

    # add any other datasets in the folder
    for name in f_names:
        with open(os.path.join(data_path, name), 'r') as file:
            frame.extend(json.load(file))

    df = pd.DataFrame(frame)
    df = df[df.path.apply(has_image)]
    df.timestamp = pd.to_datetime(df['timestamp'], unit='ms')
    df.path = df.path.str.replace('singles', 'img')

    return df
