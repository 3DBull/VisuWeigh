
import sys
#sys.path.append('../scripts')
import logging
import tensorflow as tf
import numpy as np
import os
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import json
import pandas as pd
from matplotlib.patches import Patch

LOGGER = logging.getLogger(__name__)


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
    Run model on set of images in the given pandas frame. The image frame must include a ['path'] column to the image as
     well as a ['weight'] column with the true values for the image.

    :param models: Keras compiled model
    :param image_frame: The pandas frame from which to get the X and y data to feed through the model.
    :param model_labels: The names of the models to appear on the graphics.
    :param display_each: (optional) If this is True, each image in the frame will be displayed inline along with its
     given truth and prediction.
    :param display_width: The number of columns to show images in.
    :param axis: The plot axis to attach the graphics to.
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
                        a = ax[i // display_width][i % display_width]
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
        #plt.show() # can be used in a single_threaded environment only

    # calculate the error and accuracy as an average for the set
    # print('{} average Error Rate: {:.2f}%'.format(model_labels[j], error[j]*100 / (len(image_frame) - non_existent)))
    MAA = (1 - accum_err / (len(image_frame) - non_existent)) * 100

    stats = []
    for i, label in enumerate(model_labels):
        print('{} average accuracy: {:.2f}%'.format(label, MAA[i]))

        p = error[i, :]
        stats.append({'name': label, 'mean_abs_accuracy': MAA[i], 'mean_abs_error': np.mean(np.abs(p)),
                      'error_mean': np.mean(p), 'error_std': np.std(p), 'error_min': min(p), 'error_max': max(p)})

    if display_each:
        LOGGER.info("showing images")
        return fig, stats
    else:

        return stats


def has_image(path):
    return os.path.exists(path)


def load_and_prep_data(path):
    '''
    Use this function to load training data. The data is loaded from the directory and file provided.
    Data is only kept if it has a valid image and in the range 400-1400 lb.

    :param path_to_database:
    :param dataset: Json file containing records with column names 'weight', 'timestamp', and 'path'
    :return: Pandas DataFrame
    '''
    # load training data
    with open(path, 'r') as file:
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
