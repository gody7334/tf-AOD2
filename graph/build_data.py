from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils.config import global_config
from graph.ops import image_embedding
from graph.ops import image_processing
from graph.ops import inputs as input_ops

class Build_Data(object):

    def __init__(self, mode):

        assert mode in ["train", "eval", "inference"], "mode should be 'train', 'eval', or 'inference"
        self.config = global_config.global_config
        self.mode = mode

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self.encoded_image = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.bbox_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.class_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.bbox_mask = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.class_mask = None

        self.bbox = None

        self.scale_bbox = None

        self.build_inputs()


    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, origin_bbox, classes, thread_id=0):
        """Decodes and processes an image string.

        Args:
          encoded_image: A scalar string Tensor; the encoded image.
          thread_id: Preprocessing thread id used to select the ordering of color
            distortions.

        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              origin_bbox,
                                              classes,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.
           using multithreading (enqueue and dequeue)
        Outputs:
          self.images
          self.input_seqs
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)
        """
        if self.mode == "inference":
          # In inference mode, images and inputs are fed via placeholders.
          image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
          input_feed = tf.placeholder(dtype=tf.int64,
                                      shape=[None],  # batch_size
                                      name="input_feed")

          # Process image and insert batch dimensions.
          images = tf.expand_dims(self.process_image(image_feed, input_feed), 0)
          input_seqs = tf.expand_dims(input_feed, 1)

          # No target sequences or input mask in inference mode.
          target_seqs = None
          input_mask = None
        else:
          if self.mode == "train":
            input_file_pattern = self.config.input_file_pattern
          elif self.mode == "eval":
            input_file_pattern = self.config.eval_input_file_pattern

          input_queue = input_ops.prefetch_input_data(
              self.reader,
              input_file_pattern,
              is_training=self.is_training(),
              batch_size=self.config.batch_size,
              values_per_shard=self.config.values_per_input_shard,
              input_queue_capacity_factor=self.config.input_queue_capacity_factor,
              num_reader_threads=self.config.num_input_reader_threads)

          # Image processing and random distortion. Split across multiple threads
          # with each thread applying a slightly different distortion.
          assert self.config.num_preprocess_threads % 2 == 0
          images_and_bboxs = []
          for thread_id in range(self.config.num_preprocess_threads):
            # type(serialized_sequence_example) == Tensor
            # threads that get tf record from queue
            serialized_sequence_example = input_queue.dequeue()

            # print num of element left in input_queue
            # serialized_sequence_example = tf.Print(serialized_sequence_example, data=[input_queue.size()], message="Nb elements left, queue size:")

            # get each element tensor from tensor
            encoded_image, bbox, classes = input_ops.parse_sequence_example(
                serialized_sequence_example,
                image_feature=self.config.image_feature_name,
                bbox_feature=self.config.bbox_feature_name,
                class_feature=self.config.class_feature_name)
            self.encoded_image = encoded_image
            # decode image
            image, scale_bbox, classes = self.process_image(encoded_image, bbox, classes, thread_id=thread_id)

            '''
            TODO resize bbox!!!
            '''

            # append to small batch (thread number)
            images_and_bboxs.append([image, scale_bbox, classes])

          # Batch inputs.
          queue_capacity = (2 * self.config.num_preprocess_threads *
                            self.config.batch_size)

          images, bbox_seqs, class_seqs, bbox_mask, class_mask = (
              input_ops.batch_with_dynamic_pad(images_and_bboxs,
                                              batch_size=self.config.batch_size,
                                              queue_capacity=queue_capacity))

          # images, bbox_seqs, class_seqs, bbox_mask, class_mask = (
              # input_ops.batch_with_static_pad_or_crop(images_and_bboxs,
                                              # batch_size=self.config.batch_size,
                                              # target_length=self.config.target_length,
                                              # queue_capacity=queue_capacity))

        bbox = tf.Print(bbox,[bbox])
        self.bbox = bbox
        scale_bbox = tf.Print(scale_bbox,[scale_bbox])
        self.scale_bbox = scale_bbox

        self.images = images
        self.class_seqs = class_seqs
        self.bbox_seqs = bbox_seqs
        self.class_mask = class_mask
        self.bbox_mask = bbox_mask
