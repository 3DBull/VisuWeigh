import sys
import os
import logging
import tensorflow as tf
import pickle
from VisuWeigh.lib import paths
LOGGER = logging.getLogger(__name__)


def build(name):
    '''
    Builds a keras model with the model structure layed out in this function. Copy the function and restructure to make
    a new model. The model will be pickled to a file as the output.

    :param name: The name to use for the model.
    :return: None
    '''
    try:
        # Transfer learning with a CNN Model
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                                                                         input_shape=(224, 224, 3),
                                                                         include_top=False)

        # freez the first n layers
        for layer in model.layers[:20]:
            layer.trainable = False

        # build new layers
        input_layer = tf.keras.Input(shape=(224, 224, 3))

        # use the whole model and add a couple layers
        x = model(input_layer, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(100)(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout

        bias_init = tf.keras.initializers.Constant(900)

        outputs = tf.keras.layers.Dense(1, bias_initializer=bias_init)(x)  # reduce to one node for weight prediction
        model = tf.keras.Model(input_layer, outputs, name=name)

        LOGGER.info(f'Building model with {len(model.layers)} layers')
        LOGGER.info(f'Model Summary: '
                    f'{model.summary()}')

        model.save(os.path.join(paths.MODEL_BUILDS, name))
        #with open(os.path.join(paths.MODEL_BUILDS, name+'.pkl'), 'wb') as f:
        #    pickle.dump(model, f)

    except Exception as ex:
        LOGGER.exception(f'Could not save model. {ex}')

if __name__ == "__main__":
    build(sys.argv[1])