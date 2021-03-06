# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops


import tensorflow as tf


def parse_sequence_example(serialized, image_feature, bbox_feature, class_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          bbox_feature: tf.FixedLenSequenceFeature([4], dtype=tf.float32),
          class_feature: tf.FixedLenSequenceFeature([1], dtype=tf.int64)
      }
  )

  encoded_image = context[image_feature]
  bbox = sequence[bbox_feature]
  classes = sequence[class_feature]
  return encoded_image, bbox, classes


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  # check if dataset file exist
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  """
  Queues are a convenient TensorFlow mechanism to compute tensors
  asynchronously using multiple threads. For example in the canonical
  'Input Reader' setup one set of threads generates filenames in a queue;
  a second set of threads read records from the files, processes them,
  and enqueues tensors on a second queue; a third set of threads dequeues
  these input records to construct batches and runs them through training
  operations.
  """

  if is_training:
    # create a queue to store filenames (string)
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)

    # calculate min capacity and capacity
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    # create a queue to store raw data
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    # read file (value) from filename queue
    # ps: tf reader is also tf graph operation
    _, value = reader.read(filename_queue)
    # values_queue.enqueue is a enqueue operation put value into values_queue
    enqueue_ops.append(values_queue.enqueue([value]))

  # values_queue: a queue, enqueue_ops: is a operation on values_queue
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue

def crop_pad_label(label, target_length, pad_value=0):
    '''
    crop or pad label into fix length for rnn batch training

    @ param
    label (T,?), should provide T
    target_lengt
    pad_value

    @ return
    label (target_length, ?)
    '''

    def _is_tensor(x):
        """Returns `True` if `x` is a symbolic tensor-like object.
        Args:     x: A python object to check.
        Returns:     `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
        """
        return isinstance(x, (ops.Tensor, variables.Variable))

    def max_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return math_ops.maximum(x, y)
        else:
            return max(x, y)

    def min_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return math_ops.minimum(x, y)
        else:
            return min(x, y)

    def equal_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return math_ops.equal(x, y)
        else:
            return x == y

    label = tf.cond(tf.rank(label) < 2,
            lambda: tf.expand_dims(label,1),
            lambda: tf.identity(label))

    # maybe crop
    label_length = tf.shape(label)[0]
    label = tf.slice(label, [0,0], [min_(label_length, target_length),-1])

    #maybe pad
    diff = tf.subtract(target_length,label_length)
    num_pad = max_(diff,0)
    padding = tf.stack([[0,num_pad],[0,0]])
    label = tf.pad(label,padding)

    return label


def batch_with_dynamic_pad(images_and_bboxs,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, bboxes, classes in images_and_bboxs:
    bboxes_length = tf.shape(bboxes)[0]
    bbox_indicator = tf.ones([bboxes_length,4], dtype=tf.int32)
    class_indicator = tf.ones([bboxes_length], dtype=tf.int32)
    enqueue_list.append([image, bboxes, tf.squeeze(classes,[1]), bbox_indicator, class_indicator])

  images, bbox_seqs, class_seqs, bbox_mask, class_mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(bbox_mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, bbox_seqs, class_seqs, bbox_mask, class_mask

def batch_with_static_pad_or_crop(images_and_bboxs,
                           batch_size,
                           target_length,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, bboxes, classes in images_and_bboxs:
    # Pad to fix length as attention model with rnn can(should) have fixed time step.
    bbox_seq = crop_pad_label(bboxes,target_length,0)
    class_seq = tf.squeeze(crop_pad_label(classes,target_length,-1),[1])
    bboxes_length = tf.shape(bboxes)[0]
    bbox_indicator = tf.ones([bboxes_length,4],dtype=tf.int32)
    class_indicator = tf.ones([bboxes_length], dtype = tf.int32)
    enqueue_list.append([image, bbox_seq, class_seq, bbox_indicator, class_indicator])

  images, bbox_seqs, class_seqs, bbox_mask, class_mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(bbox_mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, bbox_seqs, class_seqs, bbox_mask, class_mask
