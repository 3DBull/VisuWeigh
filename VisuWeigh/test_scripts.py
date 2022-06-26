import os
import sys
import logging
import numpy as np
import pandas as pd
from cv2 import cv2
import evaluate
import unittest
import tensorflow as tf
from VisuWeigh.lib import paths, util, weigh

LOGGER = logging.getLogger(__name__)


class TestEvaluation(unittest.TestCase):

    def test_evaluate_model_bad_model_file(self):

        # Test Case - Model Input - Keras model, but wrong architecture
        model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(3,)), tf.keras.layers.Softmax()])
        model.compile()
        model.save('_TEST_')

        with self.assertRaises(ValueError):
            util.evaluate_model(model_path='_TEST_')

        # Test Case - Different file type
        with self.assertRaises(OSError):
            util.evaluate_model(model_path=paths.EVALUATION_RESULTS)

    def test_evaluate_model_bad_data(self):
        # Test Case - Bad data in dataset
        results = util.evaluate_model(
            model_path=os.path.join(paths.TRAINED_MODELS, os.listdir(paths.TRAINED_MODELS)[0]),
            df=pd.DataFrame([{'starling': 100}]))
        self.assertIsNone(results)

    def test_evaluate_model_no_data(self):
        # Test Case - No data in dataset
        results = util.evaluate_model(
            model_path=os.path.join(paths.TRAINED_MODELS, os.listdir(paths.TRAINED_MODELS)[0]),
            df=pd.DataFrame([]))
        self.assertIsNone()


    def test_evaluate_model(self):
        # Test case - No data in dataset
        results = util.evaluate_model(model_path=os.path.join(paths.TRAINED_MODELS, os.listdir(paths.TRAINED_MODELS)[0]))
        self.assertIsInstance(results[0], dict)


class TestWeigh(unittest.TestCase):

    def test_predict_no_data(self):
        # Test case - no input
        with self.assertRaises(ValueError):
            weigh.predict()

    def test_predict_corrupt_data(self):
        # Test case - corrupted data
        ret1, ret2, ret3 = weigh.predict(images=[np.array([[[1,2],[2,2]],[[1,2],[2,2]],[[1,2],[2,2]]])])
        self.assertEqual([], ret1)
        self.assertEqual([], ret2)
        self.assertEqual([], ret3)

    def test_predict_good_data(self):
        # Test case - corrupted data
        crops, images, weights = weigh.predict(images=[cv2.imread('../docs/img_1.png')])

        self.assertIsInstance(crops, list)
        self.assertIsInstance(images, list)
        self.assertIsInstance(weights, list)

    def test_save_client_info_no_data(self):
        # Test case - no data
        ret = weigh.save_client_info(np.array([]), 'lib/test_data', 1, 23, 35)
        self.assertFalse(ret)

    def test_save_client_info_no_file(self):
        # Test case - no data
        with self.assertRaises(FileNotFoundError):
            weigh.save_client_info(cv2.imread('../docs/img_1.png'), 'phoney_filename', 1, 23, 35)

    def test_save_client_info_good(self):
        # Test case - no data
        ret = weigh.save_client_info(cv2.imread('../docs/img_1.png'), 'lib/test_data', 1, 23, 35)
        self.assertTrue(ret)

if __name__ == "__main__":
    unittest.main()
