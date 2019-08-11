from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import json
import tensorflow as tf
import cv2
import numpy as np
from absl import flags
from mnist import get_prediction

import os
import time

# pylint: disable=g-bad-import-order
from absl import app as absl_app

from tensorflow.python import eager as tfe
# pylint: enable=g-bad-import-order

from official.mnist import dataset as mnist_dataset
from official.mnist import mnist
from official.utils.flags import core as flags_core
from official.utils.misc import model_helpers
import cv2
import numpy as np


def send_msg(rdf_json):
    URL = "http://localhost:8080/greeting"
    my_json = {"a": "1"}
    PARAMS = {'json': json.dumps(my_json)}
    r = requests.get(url=URL, params=PARAMS)

    data = r.json()
    print("Got some results : ", data)


def convert_results_to_rdfJson(rslts):
    # converts the results:rslts to rdf/json format.
    return None


def define_mnist_eager_flags():
    """Defined flags and defaults for MNIST in eager mode."""
    flags_core.define_base_eager()
    flags_core.define_image()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_integer(
        name='log_interval', short_name='li', default=10,
        help=flags_core.help_wrap('batches between logging training status'))

    flags.DEFINE_string(
        name='output_dir', short_name='od', default=None,
        help=flags_core.help_wrap('Directory to write TensorBoard summaries'))

    flags.DEFINE_float(name='learning_rate', short_name='lr', default=0.01,
                       help=flags_core.help_wrap('Learning rate.'))

    flags.DEFINE_float(name='momentum', short_name='m', default=0.5,
                       help=flags_core.help_wrap('SGD momentum.'))

    flags.DEFINE_bool(name='no_gpu', short_name='nogpu', default=False,
                      help=flags_core.help_wrap(
                          'disables GPU usage even if a GPU is available'))

    flags_core.set_defaults(
        data_dir='/tmp/tensorflow/mnist/input_data',
        model_dir='../mnist_model',
        batch_size=100,
        train_epochs=10,
    )


def process_image(img, flags_obj):
    # loads a pretrained model from load_model()
    # forward pass the image using model(img) and get results
    # returns the results

    cls, prob = get_prediction(img, flags_obj)
    rslts = {"cls":cls, "prob":prob}
    return rslts


def main(_):
    # take name of input image.
    # reads the image: img
    # call process_image(img) and fetch results:rslts of a NN.
    # convert results in to rdf_json by calling function convert_results_to_rdfJson(rslts)
    # send it to server by calling send_msg(rdf_json)
    img_name = './example5.png'
    img = cv2.imread(img_name, 0)
    img = img.flatten() / 255.0
    img = np.expand_dims(img, axis=0)
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    rslts = process_image(img, flags.FLAGS)
    rslts["img_name"] = img_name
    rdf_json = convert_results_to_rdfJson(rslts)
    # send_msg(rdf_json)


if __name__ == '__main__':
    tf.enable_eager_execution()
    define_mnist_eager_flags()
    absl_app.run(main=main)

