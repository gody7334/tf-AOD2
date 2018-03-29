from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import os.path as osp
import numpy as np
import argparse
import ipdb
from datetime import datetime

project_folder = "/home/gody7334/Project/tensorflow/ipython/AOD2"

########### python arg parser
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--mode', help='mode should be one of "train" "new_train" "eval" "inference"', required=True)
parser.add_argument('-b','--checkpoint_base_dir', help='checkpoint base dir', required=False)
parser.add_argument('-e','--checkpoint_sub_dir', help='experiement name, format: YYYYMMDD-HHmmSS-TAG', required=False)
parser.add_argument('-d','--device', help='cpu, if provided', required=False)
parser.add_argument('-l','--log', help='t, if provided', required=False)
parser.add_argument('-g','--git', help='t, if provided', required=False)
args = vars(parser.parse_args())

########### tf arg parser
parse_args = tf.app.flags.FLAGS

# tf.flags.DEFINE_string("input_file_pattern", project_folder+"/data/dataset/train-?????-of-00256", "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("input_file_pattern", project_folder+"/data/dataset/train-000??-of-00256", "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", project_folder+"/data/pretrain_model/inception_v3.ckpt", "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", project_folder+"/data/check_point", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False, "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10, "Frequency at which loss and global step are logged.")

# evaluation
tf.flags.DEFINE_string("eval_input_file_pattern", project_folder+"/data/dataset/val-?????-of-00004", "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("trained_model_checkpoint_dir", project_folder+"/data/check_point", "Directory for saving and loading trained model checkpoints.")
tf.flags.DEFINE_string("eval_dir", project_folder+"/data/eval_check_point", "Directory for saving and loading evaluation checkpoints.")

# log
tf.flags.DEFINE_string("log_dir", project_folder+"/data/log", "Directory for log tensor.")
tf.logging.set_verbosity(tf.logging.INFO)

global_config = None
def assign_config():
    global global_config
    global_config = Global_Config()


class Global_Config(object):

    def __init__(self):
        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        def mkdir(d):
            if not os.path.exists(d):
                os.makedirs(d)

        # new checkpoint structure
        self.checkpoint_base_dir = project_folder+"/data/check_point/";
        if args['checkpoint_base_dir'] is not None:
            self.checkpoint_base_dir = args['checkpoint_base_dir']
        self.checkpoint_sub_dir = now + '/'
        if args['checkpoint_sub_dir'] is not None:
            self.checkpoint_sub_dir = args['checkpoint_sub_dir'] + '/'
        self.tf_model_dir = self.checkpoint_base_dir + self.checkpoint_sub_dir + "model/"
        self.tf_log_dir = self.checkpoint_base_dir + self.checkpoint_sub_dir + "tf_log/"
        self.tb_train_log_dir = self.checkpoint_base_dir + self.checkpoint_sub_dir + "tb_train/"
        self.tb_eval_log_dir = self.checkpoint_base_dir + self.checkpoint_sub_dir + "tb_eval/"
        self.tb_test_log_dir = self.checkpoint_base_dir + self.checkpoint_sub_dir + "tb_test/"

        mkdir(self.checkpoint_base_dir+self.checkpoint_sub_dir)
        mkdir(self.tf_model_dir)
        mkdir(self.tf_log_dir)
        mkdir(self.tb_train_log_dir)
        mkdir(self.tb_eval_log_dir)
        mkdir(self.tb_test_log_dir)

        self.mode = args['mode']
        self.device = args['device']
        self.is_tf_log = False
        if args['log'] == "t":
            self.is_tf_log=True

        self.log_every_n_steps = 500

        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be pmrovided in training and evaluation modes.
        self.input_file_pattern = None

        #training directory.
        self.train_dir = self.tb_train_log_dir
        self.train_checkpoint_log_dir = self.tf_model_dir

        #log directory.
        self.log_dir = self.tf_log_dir

        # Image format ("jpeg" or "png").
        self.image_format = "jpeg"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1

        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/caption_ids"
        # Name of the SequenceExample feature list containing integer bbox.
        self.bbox_feature_name = "image/bbox"
        # Name of the SequenceExample feature list containing integer class.
        self.class_feature_name = "image/category"



        # Number of unique words in the vocab (plus 1, for <UNK>).
        # The default value is larger than the expected actual vocab size to allow
        # for differences between tokenizer versions used in preprocessing. There is
        # no harm in using a value greater than the actual vocab size, but using a
        # value less than the actual vocab size will result in an error.
        self.vocab_size = 12000

        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Batch size.
        self.batch_size = 20

        # Target length (max time steps)
        self.target_length = 20

        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        # self.num_examples_per_epoch = 586363
        self.num_examples_per_epoch = 20000

        # Optimizer for training the model.
        self.optimizer = "SGD"
        # self.optimizer = "Adam"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 1.0e-1
        self.learning_rate_decay_factor = 1.0
        self.num_epochs_per_decay = 1000.0

        # Learning rate when fine tuning the Inception v3 parameters.
        self.train_inception_learning_rate = 0.0005

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 3

        # File containing an Inception v3 checkpoint to initialize the variables
        # of the Inception model. Must be provided when starting training for the
        # first time.
        self.inception_checkpoint_file = None

        # Dimensions of Inception v3 input images.
        self.image_height = 299
        self.image_width = 299

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # endpoints_shapes = {'Conv2d_1a_3x3': [batch_size, 149, 149, 32],
        #             'Conv2d_2a_3x3': [batch_size, 147, 147, 32],
        #             'Conv2d_2b_3x3': [batch_size, 147, 147, 64],
        #             'MaxPool_3a_3x3': [batch_size, 73, 73, 64],
        #             'Conv2d_3b_1x1': [batch_size, 73, 73, 80],
        #             'Conv2d_4a_3x3': [batch_size, 71, 71, 192],
        #             'MaxPool_5a_3x3': [batch_size, 35, 35, 192],
        #             'Mixed_5b': [batch_size, 35, 35, 256],
        #             'Mixed_5c': [batch_size, 35, 35, 288],
        #             'Mixed_5d': [batch_size, 35, 35, 288],
        #             'Mixed_6a': [batch_size, 17, 17, 768],
        #             'Mixed_6b': [batch_size, 17, 17, 768],
        #             'Mixed_6c': [batch_size, 17, 17, 768],
        #             'Mixed_6d': [batch_size, 17, 17, 768],
        #             'Mixed_6e': [batch_size, 17, 17, 768],
        #             'Mixed_7a': [batch_size, 8, 8, 1280],
        #             'Mixed_7b': [batch_size, 8, 8, 2048],
        #             'Mixed_7c': [batch_size, 8, 8, 2048]}

        self.image_embedding_depth = 512 #Mixed_6e 17*17
        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 512
        self.num_lstm_units = 1024

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        # assign parse argument to configuration object
        self.assign_global_config();

        # Root directory of project
        self.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
        # Data directory
        self.DATA_DIR = osp.abspath(osp.join(self.ROOT_DIR, 'data'))

        # Name (or path to) the matlab executable
        self.MATLAB = 'matlab'

        # Interval between evaluation runs.
        self.eval_interval_secs = 120

        # Number of examples for evaluation.
        self.num_eval_examples = 100

        # Minimum global step to run evaluation.
        self.min_global_step = 100

        # Number of classes in the dataset for classification task
        self.num_classes = 100 # 91 in mscoco

    def assign_global_config(self):
        # assert parse_args.train_dir, "--train_dir is required"
        # # Create training directory.
        # self.train_dir=parse_args.train_dir
        # if not tf.gfile.IsDirectory(self.train_dir):
            # tf.logging.info("Creating training directory: %s", train_dir)
            # tf.gfile.MakeDirs(train_dir)

        # assert parse_args.log_dir, "--log_dir is required"
        # # Create log directory.
        # self.log_dir=parse_args.log_dir
        # if not tf.gfile.IsDirectory(self.log_dir):
            # tf.logging.info("Creating training directory: %s", log_dir)
            # tf.gfile.MakeDirs(log_dir)

        # # Evaluate training directory.
        # self.eval_dir=parse_args.eval_dir
        # if not tf.gfile.IsDirectory(self.eval_dir):
            # tf.logging.info("Creating training directory: %s", eval_dir)
            # tf.gfile.MakeDirs(eval_dir)

        assert parse_args.input_file_pattern, "--input_file_pattern is required"
        self.input_file_pattern=parse_args.input_file_pattern

        assert parse_args.inception_checkpoint_file, "--inception_checkpoint_file is required"
        self.inception_checkpoint_file=parse_args.inception_checkpoint_file

        assert parse_args.eval_input_file_pattern, "--eval_input_file_pattern is required"
        self.eval_input_file_pattern=parse_args.eval_input_file_pattern

        assert parse_args.trained_model_checkpoint_dir, "--trained_model_checkpoint_dir is required"
        self.trained_model_checkpoint_dir=parse_args.trained_model_checkpoint_dir


