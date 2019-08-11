# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

This program demonstrates training of the convolutional neural network model
defined in mnist.py with eager execution enabled.

If you are not interested in eager execution, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from official.mnist import dataset as mnist_dataset
from official.mnist import mnist
from official.utils.misc import model_helpers




def get_prediction(img, flags_obj):
    """Run MNIST training and eval loop in eager mode.

    Args:
      flags_obj: An object containing parsed flag values.
    """

    # model_helpers.apply_clean(flags.FLAGS)
    model_helpers.apply_clean(flags_obj)

    # Automatically determine device and data_format
    (device, data_format) = ('/gpu:0', 'channels_first')
    if flags_obj.no_gpu or not tf.test.is_gpu_available():
        (device, data_format) = ('/cpu:0', 'channels_last')
    # If data_format is defined in FLAGS, overwrite automatically set value.
    if flags_obj.data_format is not None:
        data_format = flags_obj.data_format
    print('Using device %s, and data format %s.' % (device, data_format))

    # Load the datasets
    # train_ds = mnist_dataset.train(flags_obj.data_dir).shuffle(60000).batch(
    #     flags_obj.batch_size)
    test_ds = mnist_dataset.test(flags_obj.data_dir).batch(
        flags_obj.batch_size)

    # Create the model and optimizer
    model = mnist.create_model(data_format)
    optimizer = tf.train.MomentumOptimizer(flags_obj.lr, flags_obj.momentum)

    # Create file writers for writing TensorBoard summaries.
    if flags_obj.output_dir:
        # Create directories to which summaries will be written
        # tensorboard --logdir=<output_dir>
        # can then be used to see the recorded summaries.
        train_dir = os.path.join(flags_obj.output_dir, 'train')
        test_dir = os.path.join(flags_obj.output_dir, 'eval')
        tf.gfile.MakeDirs(flags_obj.output_dir)
    else:
        train_dir = None
        test_dir = None



    # Create and restore checkpoint (if one exists on the path)
    checkpoint_prefix = os.path.join(flags_obj.model_dir, 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step_counter=step_counter)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(tf.train.latest_checkpoint(flags_obj.model_dir))

    # Train and evaluate for a set number of epochs.
    with tf.device(device):


        logits = model(img, training=False)
        prob = tf.nn.softmax(logits)
        cls = tf.argmax(logits, axis=1, output_type=tf.int64)

        return int(cls.numpy()[0]), prob.numpy()

