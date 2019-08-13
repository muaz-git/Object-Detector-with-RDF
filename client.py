from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from absl import flags
import sys
from rdflib.namespace import DC, FOAF

tensorflow_path = r'C:\\Users\\mumu01\\AppData\\Local\\Continuum\\miniconda2\\envs\\python361envgpu\\Lib\\site-packages\\tensorflow'
mdls_path = tensorflow_path + '\\models'

if not os.path.isdir(mdls_path):
    print(
        "To run this code successfully please copy 'models' folder in your tensorflow_path directory. It can be downloaded from https://github.com/tensorflow/models")
    print("tensorflow_path is relative to user's system, so please change the variable:tensorflow_path accordingly")
    exit()

sys.path.append(mdls_path)
from mnist import get_prediction
from absl import app as absl_app

from official.utils.flags import core as flags_core
import cv2
import numpy as np
import uuid
from rdflib import URIRef, Literal, Graph

import shutil

tmp_dir = './tmp/'
if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

f_name = tmp_dir + str(uuid.uuid4())
url = 'http://localhost:8080/greeting'

mnist_saved_model = '../mnist_model'

print("\n\n\tPlease make sure that pre-trained model is saved in directory: " + mnist_saved_model)


def send_msg():
    files = {'file': open(f_name, 'rb')}
    r = requests.post(url, files=files)

    data = r.json()


def save_as_rdfxml(rslts):
    # saves the results:rslts to xml file.
    g = Graph()

    hst = "http://example.org/"
    img_dest = URIRef(hst + rslts["img_name"])

    # img_name = hst+rslts["img_name"]
    # pred = URIRef(img_name + '/pred')
    # uri = URIRef(img_name)
    # val = URIRef(hst+str(rslts['cls']))

    g.add((img_dest, FOAF.pred, Literal(rslts['cls'])))

    # g.add((uri, pred, val))

    g.serialize(destination=f_name, format='xml')


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
        model_dir=mnist_saved_model,
        batch_size=100,
        train_epochs=10,
    )


def process_image(img, flags_obj):
    # loads a pretrained model from load_model()
    # forward pass the image using model(img) and get results
    # returns the results

    cls, prob = get_prediction(img, flags_obj)
    rslts = {"cls": cls, "prob": prob}
    return rslts


def main(_):
    # take name of input image.
    img_name = input("\n\nEnter path to image: ")
    if not os.path.exists(img_name):
        img_name = 'example3.png'
        print("\n\tPath does not exist. Loading default file i.e. " + img_name)

    print("\nLoading " + img_name)
    # exit()

    # reads the image: img

    img = cv2.imread(img_name, 0)
    # preprocessing for the model.

    img = img.flatten() / 255.0
    img = np.expand_dims(img, axis=0)
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # call process_image(img) and fetch results:rslts of a NN.
    rslts = process_image(img, flags.FLAGS)
    rslts["img_name"] = img_name
    # convert results in to rdf_xml by calling function save_as_rdfxml(rslts)
    save_as_rdfxml(rslts)
    # send it to server by calling send_msg(rdf_json)
    send_msg()


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    define_mnist_eager_flags()
    absl_app.run(main=main)
