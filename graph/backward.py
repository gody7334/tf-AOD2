from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils.config import global_config

class Backward(object):

    def __init__(self, model):
        self.model = model
        self.train_inception = global_config.parse_args.train_inception
        self.training_config = global_config.global_config
        self.model_config = global_config.global_config

        self.train_op = None

        self.build_optimization()

    def build_optimization(self):
        # Set up the learning rate.
        learning_rate_decay_fn = None
        if self.train_inception:
            learning_rate = tf.constant(self.training_config.train_inception_learning_rate)
        else:
            learning_rate = tf.constant(self.training_config.initial_learning_rate)
            if self.training_config.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (self.training_config.num_examples_per_epoch /
                                     self.model_config.batch_size)
                decay_steps = int(num_batches_per_epoch *
                              self.training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=self.training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=self.model.total_loss,
            global_step=self.model.global_step,
            learning_rate=learning_rate,
            optimizer=self.training_config.optimizer,
            clip_gradients=self.training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        self.train_op = train_op

