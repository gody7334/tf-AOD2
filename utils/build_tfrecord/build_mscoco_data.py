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
"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...

and

  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:

  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256

and

  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004

and

  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008

Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace

  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.

NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.

Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading
# import cPickle as pickle
import dill as pickle

from pprint import pprint as pp
import colored_traceback.always

import nltk.tokenize
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("train_image_dir", "/tmp/train2014/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/tmp/val2014",
                       "Validation image directory.")

tf.flags.DEFINE_string("train_captions_file", "/tmp/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "/tmp/captions_val2014.json",
                       "Validation captions JSON file.")

tf.flags.DEFINE_string("train_captions_pickle", "/dataset/mscoco/raw-data/annotations_pickle/captions_train2014.pickle",
                       "Training captions pickle file.")
tf.flags.DEFINE_string("val_captions_pickle", "/dataset/mscoco/raw-data/annotations_pickle/captions_val2014.pickle",
                       "Validation captions pickle file.")

tf.flags.DEFINE_string("train_bbox_pickle", "/dataset/mscoco/raw-data/annotations_pickle/bbox_train2014.pickle",
                       "Training captions pickle file.")
tf.flags.DEFINE_string("val_bbox_pickle", "/dataset/mscoco/raw-data/annotations_pickle/bbox_val2014.pickle",
                       "Validation captions pickle file.")

tf.flags.DEFINE_string("train_instances_file", "/tmp/instances_train2014.json",
                       "Training instances JSON file.")
tf.flags.DEFINE_string("val_instances_file", "/tmp/instances_val2014.json",
                       "Validation instances JSON file.")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_bbox", [-1000,-1000,-1000,-1000],
                       "Special array added to the beginning of each bboxs.")
tf.flags.DEFINE_string("end_bbox", [-2000,-2000,-2000,-2000],
                       "Special array added to the end of each bboxs.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/tmp/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 1,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

ImageBboxdata = namedtuple("ImageBboxdata",
                           ["image_id", "filename", "categories", "bboxs", "segmentations"])
# ImageBboxdata.__module__ = "ImageBboxdata"

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])
# ImageMetadata.__module__ = "ImageMetadata"

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _float_feature(value):
  """Wrapper for inserting an float Feature into a SequenceExample proto."""
  if isinstance(value, list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  else:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
  """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _to_sequence_example(image, decoder, vocab, label_type = "caption"):
  """Builds a SequenceExample proto for an image-caption pair.

  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.

  Returns:
    A SequenceExample proto.
  """
#   import ipdb; ipdb.set_trace()
  with tf.gfile.FastGFile(image.filename, "r") as f:
    encoded_image = f.read()

  try:
    # here only check jpg format using decoder,
    # but write ecoded image into tf record
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return

  context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
  })

  if label_type == "caption":
      assert len(image.captions) == 1
      caption = image.captions[0]
      caption_ids = [vocab.word_to_id(word) for word in caption]
      feature_lists = tf.train.FeatureLists(feature_list={
          "image/caption": _bytes_feature_list(caption),
          "image/caption_ids": _int64_feature_list(caption_ids)
      })
      sequence_example = tf.train.SequenceExample(
          context=context, feature_lists=feature_lists)
  elif label_type == "bbox":
      assert len(image.bboxs) == 1
      category = image.categories[0]
      bbox = image.bboxs[0]
      segmentation = image.segmentations[0]

      # segmentation label has some problem, deal with it later
      feature_lists = tf.train.FeatureLists(feature_list={
          "image/category": _int64_feature_list(category),
          "image/bbox": _float_feature_list(bbox)
          # "image/segmentation": _float_feature_list(segmentation)
      })
      sequence_example = tf.train.SequenceExample(
          context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards, label_type = "caption"):
  """Processes and saves a subset of images as TFRecord files in one thread.

  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).

#   import ipdb; ipdb.set_trace()
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):
    # import ipdb; ipdb.set_trace()
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in images_in_shard:
      image = images[i]

      sequence_example = _to_sequence_example(image, decoder, vocab, label_type)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote %d image-%s pairs to %s" %
          (datetime.now(), thread_index, shard_counter, label_type, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-%s pairs to %d shards." %
        (datetime.now(), thread_index, counter, label_type, num_shards_per_batch))
  sys.stdout.flush()

def _process_dataset(name, images, vocab, num_shards, label_type = "caption"):
  """Processes a complete data set and saves it as a TFRecord.

  Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Break up each image into a separate entity for each caption.
  import ipdb; ipdb.set_trace()

  if label_type == "caption":
    images = [ImageMetadata(image.image_id, image.filename, [caption])
                for image in images for caption in image.captions]
  elif label_type == "bbox":
    images = [ImageBboxdata(image.image_id, image.filename, [category], [bbox], [segmentation])
                for image in images for category in image.categories for bbox in image.bboxs for segmentation in image.segmentations]

  # Shuffle the ordering of images. Make the randomization repeatable.
  random.seed(12345)
  random.shuffle(images)

  # Break the images into num_threads batches. Batch i is defined as
  # images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a utility for decoding JPEG images to run sanity checks.
  decoder = ImageDecoder()

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in xrange(len(ranges)):
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards, label_type)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _create_vocab(captions):
  """Creates the vocabulary of word to word_id.

  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.

  Args:
    captions: A list of lists of strings.

  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab

def _process_bbox(bbox):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.

  Returns:
    A list of strings; the tokenized caption.
  """
  import random
  bboxs_array = []
  for i in range(5):
    bboxs = [FLAGS.start_bbox]
    bboxs.extend(random.sample(bbox, len(bbox)))
    bboxs.append(FLAGS.end_bbox)
    bboxs_array.append(bboxs)

  return bboxs_array

def _process_category_bbox_segmentation(category, bbox, segmentation):
  category_array = []
  bbox_array = []
  segmentation_array = []
  category_array.append(category)
  bbox_array.append(bbox)
  segmentation_array.append(segmentation)
  return (category_array, bbox_array, segmentation)

def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.

  Returns:
    A list of strings; the tokenized caption.
  """
  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(FLAGS.end_word)
  return tokenized_caption


def _load_and_process_metadata(captions_file, instances_file, image_dir):
  """Loads image metadata from a JSON file and processes the captions.

  Args:
    captions_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.

  Returns:
    A list of ImageMetadata.
  """
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)

  with tf.gfile.FastGFile(instances_file, "r") as f:
    instance_data = json.load(f)

  # Extract the filenames.
  id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

  id_to_filename_inst = [(x["id"], x["file_name"]) for x in instance_data["images"]]

  # Extract the instances. Each image_id is associated with multiple captions.
  id_to_bbox = {}
  id_to_category = {}
  id_to_segmentation = {}
  for annotation in instance_data["annotations"]:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category =annotation['category_id']
    segmentation = annotation['segmentation']
    id_to_bbox.setdefault(image_id, [])
    id_to_bbox[image_id].append(bbox)
    id_to_category.setdefault(image_id, [])
    id_to_category[image_id].append(category)
    id_to_segmentation.setdefault(image_id, [])
    id_to_segmentation[image_id].append(segmentation)

  # remove missing annotation in file name
  image_set = set([x[0] for x in id_to_filename_inst])
  annotation_set = set(id_to_bbox.keys())
  miss_annotation_set = image_set - annotation_set
  id_to_filename_inst_clone = list(id_to_filename_inst)
  for id_file in id_to_filename_inst_clone:
    if id_file[0] in miss_annotation_set:
      id_to_filename_inst.remove(id_file)

  assert len(id_to_filename_inst) == len(id_to_bbox)
  assert set([x[0] for x in id_to_filename_inst]) == set(id_to_bbox.keys())
  assert len(id_to_filename_inst) == len(id_to_category)
  assert set([x[0] for x in id_to_filename_inst]) == set(id_to_category.keys())
  assert len(id_to_filename_inst) == len(id_to_segmentation)
  assert set([x[0] for x in id_to_filename_inst]) == set(id_to_segmentation.keys())

  print("Loaded bbox metadata for %d images from %s" %
        (len(id_to_filename_inst), instances_file))

  # Extract the captions. Each image_id is associated with multiple captions.
  id_to_captions = {}
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])
    id_to_captions[image_id].append(caption)

  assert len(id_to_filename) == len(id_to_captions)
  assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
  print("Loaded caption metadata for %d images from %s" %
        (len(id_to_filename), captions_file))

  # Process the bbox and combine the data into a list of ImageBboxdata.
  print("Processing bbox.")
  image_metadata_bbox = []
  num_bbox = 0
  for image_id, base_filename in id_to_filename_inst:
    filename = os.path.join(image_dir, base_filename)
    bboxs = _process_bbox(id_to_bbox[image_id])
    categories, bboxs, segmentations = _process_category_bbox_segmentation(
                                          id_to_category[image_id],
                                          id_to_bbox[image_id],
                                          id_to_segmentation[image_id])
    image_metadata_bbox.append(ImageBboxdata(image_id, filename, categories, bboxs, segmentations))
    num_bbox += len(bboxs)
  print("Finished processing %d bboxs for %d images in %s" %
        (num_bbox, len(id_to_filename), instances_file,))

  # Process the captions and combine the data into a list of ImageMetadata.
  print("Processing captions.")
  image_metadata = []
  num_captions = 0
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(image_dir, base_filename)
    captions = [_process_caption(c) for c in id_to_captions[image_id]]
    image_metadata.append(ImageMetadata(image_id, filename, captions))
    num_captions += len(captions)
  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))

#   return image_metadata_bbox
  return image_metadata, image_metadata_bbox


def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Load image metadata from caption files.
  # try:
    # mscoco_train_dataset = pickle.load(open(FLAGS.train_captions_pickle, "rb"))
    # mscoco_bbox_train_dataset = pickle.load(open(FLAGS.train_bbox_pickle, "rb"))
  # except (OSError, IOError) as e:
    # mscoco_train_dataset, mscoco_bbox_train_dataset = _load_and_process_metadata(
                                                    # FLAGS.train_captions_file,
                                                    # FLAGS.train_instances_file,
                                                    # FLAGS.train_image_dir)
    # pickle.dump(mscoco_train_dataset, open(FLAGS.train_captions_pickle, "wb"))
    # pickle.dump(mscoco_bbox_train_dataset, open(FLAGS.train_bbox_pickle, "wb"))

  # try:
    # mscoco_val_dataset = pickle.load(open(FLAGS.val_captions_pickle, "rb"))
    # mscoco_bbox_val_dataset = pickle.load(open(FLAGS.val_bbox_pickle, "rb"))
  # except (OSError, IOError) as e:
    # mscoco_val_dataset, mscoco_bbox_val_dataset = _load_and_process_metadata(
                                                  # FLAGS.val_captions_file,
                                                  # FLAGS.val_instances_file,
                                                  # FLAGS.val_image_dir)
    # pickle.dump(mscoco_val_dataset, open(FLAGS.val_captions_pickle, "wb"))
    # pickle.dump(mscoco_bbox_val_dataset, open(FLAGS.val_bbox_pickle, "wb"))

  # mscoco_train_dataset = mscoco_train_dataset[:1000]
  # mscoco_val_dataset = mscoco_val_dataset[:1000]
  # mscoco_bbox_train_dataset = mscoco_bbox_train_dataset[:1000]
  # mscoco_bbox_val_dataset = mscoco_bbox_val_dataset[:1000]

  # pickle.dump(mscoco_train_dataset, open(FLAGS.train_captions_pickle+".1000", "wb"))
  # pickle.dump(mscoco_bbox_train_dataset, open(FLAGS.train_bbox_pickle+".1000", "wb"))
  # pickle.dump(mscoco_val_dataset, open(FLAGS.val_captions_pickle+".1000", "wb"))
  # pickle.dump(mscoco_bbox_val_dataset, open(FLAGS.val_bbox_pickle+".1000", "wb"))

# load whole dataset
  # mscoco_train_dataset = pickle.load(open(FLAGS.train_captions_pickle, "rb"))
  # mscoco_bbox_train_dataset = pickle.load(open(FLAGS.train_bbox_pickle, "rb"))
  # mscoco_val_dataset = pickle.load(open(FLAGS.val_captions_pickle, "rb"))
  # mscoco_bbox_val_dataset = pickle.load(open(FLAGS.val_bbox_pickle, "rb"))

# load tiny dataset
  mscoco_train_dataset = pickle.load(open(FLAGS.train_captions_pickle+".1000", "rb"))
  mscoco_bbox_train_dataset = pickle.load(open(FLAGS.train_bbox_pickle+".1000", "rb"))
  mscoco_val_dataset = pickle.load(open(FLAGS.val_captions_pickle+".1000", "rb"))
  mscoco_bbox_val_dataset = pickle.load(open(FLAGS.val_bbox_pickle+".1000", "rb"))

  # Redistribute the MSCOCO data as follows:
  #   train_dataset = 100% of mscoco_train_dataset + 85% of mscoco_val_dataset.
  #   val_dataset = 5% of mscoco_val_dataset (for validation during training).
  #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).

  train_bbox_cutoff = int(0.85 * len(mscoco_bbox_val_dataset))
  val_bbox_cutoff = int(0.90 * len(mscoco_bbox_val_dataset))
  train_bbox_dataset = mscoco_bbox_train_dataset + mscoco_bbox_val_dataset[0:train_bbox_cutoff]
  val_bbox_dataset = mscoco_bbox_val_dataset[train_bbox_cutoff:val_bbox_cutoff]
  test_bbox_dataset = mscoco_bbox_val_dataset[val_bbox_cutoff:]

  import ipdb; ipdb.set_trace()

  _process_dataset("train", train_bbox_dataset, None, FLAGS.train_shards, label_type = "bbox")
  _process_dataset("val", val_bbox_dataset, None, FLAGS.val_shards, label_type = "bbox")
  _process_dataset("test", test_bbox_dataset, None, FLAGS.test_shards, label_type = "bbox")

#train_cutoff = int(0.85 * len(mscoco_val_dataset))
#   val_cutoff = int(0.90 * len(mscoco_val_dataset))
#   train_dataset = mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
#   val_dataset = mscoco_val_dataset[train_cutoff:val_cutoff]
#   test_dataset = mscoco_val_dataset[val_cutoff:]

#   # Create vocabulary from the training captions.
#   train_captions = [c for image in train_dataset for c in image.captions]
#   vocab = _create_vocab(train_captions)

#   _process_dataset("train", train_dataset, vocab, FLAGS.train_shards, label_type = "caption")
#   _process_dataset("val", val_dataset, vocab, FLAGS.val_shards, label_type = "caption")
#   _proc   ess_dataset("test", test_dataset, vocab, FLAGS.test_shards, label_type = "caption")


if __name__ == "__main__":
  tf.app.run()
