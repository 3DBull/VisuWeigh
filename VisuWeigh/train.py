"""
VisuWeigh Project
This is the training script to train the entire directory of built models.
Each model in the directory will be fit to the data.
Input models should be full keras models.


Derek Syme
2022-06-02
"""
import os
import logging
import subprocess
from IPython import get_ipython
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tensorboard import program
import datetime
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from VisuWeigh.lib.generator import Generator
from importlib import reload
from VisuWeigh.lib.util import test_on_image, load_and_prep_data
from VisuWeigh.lib import paths
from VisuWeigh.models import *
import pickle
import pandas as pd

LOGGER = logging.getLogger(__name__)

### SETTINGS ####
SELECT_GPU = 0  # zero to use default
HOST = '192.168.0.179 '
PORT = 6005
START_TENSORBOARD = True  # only compatible with windows os
#################


# fix some memory expansion issues by allowing growth on the gpu
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')

LOGGER.info(f'Available GPUs: {gpus}')

if SELECT_GPU:
    tf.config.experimental.set_visible_devices(gpus[SELECT_GPU - 1], 'GPU')


def start_tensorboard():

    subprocess.call(f'tensorboard --logdir={paths.TRAINING_LOGS} --host={HOST} --port={PORT}')
    url = f'http://{HOST}:{PORT}'
    print(f"Tensorboard listening on {url}")


def load_model(name: str) -> tf.keras.Model:
    """
    Loads a pickled model from the database and returns it as a keras model.

    :param name: the name of the model (with .pkl extension)
    :return: A keras model
    """
    model = tf.keras.models.load_model(os.path.join(paths.MODEL_BUILDS, name))
    return model


def train_test_split(d: pd.DataFrame, frac: float = 0.2) -> tuple:
    """
    Split the training data into training and testing
    :param d: The dataset to operate on
    :param frac: The fraction of the data to return as a test dataframe

    :return: tuple of DataFrames: (train, test)
    """
    test = d.sample(frac=frac, axis=0)
    train = d.drop(index=test.index).sample(frac=1, axis=0)
    return train, test


def create_gen(training_df, validation_df, model_input_shape, color='grayscale', batch_size=32):
    """
    Create a generator with flow from directory capabilities.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                            )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=training_df,
        directory=paths.TRAINING_IMAGES,
        x_col='path',
        y_col='weight',
        target_size=model_input_shape,
        batch_size=batch_size,
        class_mode='raw',
        color_mode=color,
        shuffle=True,
    )

    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=paths.TRAINING_IMAGES,
        x_col='path',
        y_col='weight',
        target_size=model_input_shape,
        batch_size=batch_size,
        class_mode='raw',
        color_mode=color,
        shuffle=True,
    )

    return train_generator, validation_generator


def staged_freeze_training_model(model, model_name, train_data, val_data):

    # Set up parameters
    train_config = paths.config['training']
    steps_per_epoch = train_config['steps_per_epoch']
    opt_name = train_config['opt']
    val_steps = train_config['val_steps']

    if train_config['cos_anneal']:
        # Use cosine decay optimizer
        first_decay_steps = 10 * steps_per_epoch
        t_mul = 2
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, first_decay_steps, t_mul=t_mul)
    else:
        lr_decayed_fn = 0.001

    if opt_name == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr_decayed_fn)

    elif opt_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)

    else:
        LOGGER.warning(f'Invalid or no optimizer option provided. Defaulting to Adam.')
        opt = 'adam'

    # Setup the saving checkpoints
    save_name = '{}_{}_{}_{}'.format(model_name, opt_name, "{epoch}", "{val_loss}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(paths.TRAINED_MODELS, save_name),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Setup tensorboard
    logs_base_dir = paths.TRAINING_LOGS
    logdir = os.path.join(logs_base_dir, model.name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # freeze most of the model
    model.trainable = True
    freeze = train_config['fit1']['freeze_layers']
    if freeze[1] > freeze[0]:
        for layer in model.layers[freeze[0]: freeze[1]]:
            layer.trainable = False

    # Compile the model
    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        metrics=["mean_squared_error"])

    # train to initialize the new weights
    model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=train_config['fit1']['to_epoch'],
        initial_epoch=0,
        validation_data=val_data,
        validation_steps=val_steps,
        callbacks=[tensorboard_callback, checkpoint_callback],
        # verbose=0,
    )

    # unfreeze most of the model and train
    model.trainable = True
    freeze = train_config['fit2']['freeze_layers']
    if freeze[1] > freeze[0]:
        for layer in model.layers[freeze[0]: freeze[1]]:
            layer.trainable = False

    # recompile
    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        metrics=["mean_squared_error"])

    # train again
    model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=train_config['fit2']['to_epoch'],
        initial_epoch=train_config['fit1']['to_epoch'],
        validation_data=val_data,
        validation_steps=val_steps,
        callbacks=[tensorboard_callback, checkpoint_callback],
        verbose=0,
    )

    # unfreeze the rest of the model
    model.trainable = True
    freeze = train_config['fit3']['freeze_layers']
    if freeze[1] > freeze[0]:
        for layer in model.layers[freeze[0]: freeze[1]]:
            layer.trainable = False

    # recompile
    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        metrics=["mean_squared_error"])

    # train again
    model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=train_config['fit3']['to_epoch'],
        initial_epoch=train_config['fit2']['to_epoch'],
        validation_data=val_data,
        validation_steps=val_steps,
        callbacks=[tensorboard_callback, checkpoint_callback],
        verbose=0,
    )

    return model


def main():
    '''
    Trains all the models in the database on all the training data in the database.
    '''
    try:

        if START_TENSORBOARD:
            start_tensorboard()

        # Load the training data
        df = load_and_prep_data(os.path.join(paths.TRAINING_DATA, paths.DATASET))

        # SPLIT DATA
        # Reserve a set data period for evaluation. This ensures that no cows in the training data show up in the evaluation data.
        cuttoff = '2022-03-13'
        cutoff2 = '2022-03-31'
        eval_df = df[(df.timestamp > cuttoff) & (df.timestamp < cutoff2)].copy()
        training_df = df[(df.timestamp < cuttoff) | (df.timestamp > cutoff2)].copy()

        # Split training into training and testing sets
        train_df, test_df = train_test_split(training_df)
        LOGGER.info(f'Split data into {len(training_df)} training points and {len(eval_df)} evaluation points')

        # Load the generator
        train_generator, val_generator = create_gen(
            train_df,
            test_df,
            (224, 224),
            color='rgb',
            batch_size=paths.config['training']['batch_size']
        )

        # Get model names
        model_names = os.listdir(paths.MODEL_BUILDS)

        # TRAINING LOOP
        for model_name in model_names:

            print(f'training with {model_name}')

            # get model
            model = load_model(model_name)

            # Train the model
            model = staged_freeze_training_model(model, model_name, train_generator, val_generator)

            # Save the model
            # This is unnecessary if using the save best callback
            #model.save(os.path.join(paths.TRAINED_MODELS, model_name[:-4]))

    except Exception as ex:
        LOGGER.exception(f'An exception occurred while training: {ex}')
        exit(1)



if __name__ == "__main__":
    main()
