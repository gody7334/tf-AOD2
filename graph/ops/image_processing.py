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

"""Helper functions for image preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def distort_image(image, thread_id):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
#   with tf.name_scope("flip_horizontal", values=[image]):
#     image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  color_ordering = thread_id % 2
  with tf.name_scope("distort_color", values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

def bbox_tranformation_xywh_yxyx(mscoco_bboxes, name="bbox_trans_xywh_yxyx"):
  with tf.name_scope(name):
      # mscoco def (top-left is min)
      xmin = tf.slice(mscoco_bboxes,[0,0,0],[-1,-1,1])
      ymin = tf.slice(mscoco_bboxes,[0,0,1],[-1,-1,1])
      xmax = tf.add(xmin,tf.slice(mscoco_bboxes,[0,0,2],[-1,-1,1]))
      ymax = tf.add(ymin,tf.slice(mscoco_bboxes,[0,0,3],[-1,-1,1]))
      # tf bbox coor
      tf_bbox = tf.concat([ymin, xmin, ymax, xmax],2)
  return tf_bbox

def draw_tf_bounding_boxes(images, scale_bboxes, name=""):
  # #mscoco def (top-left is min)
  # xmin = tf.slice(scale_bboxes,[0,0,0],[-1,-1,1])
  # ymin = tf.slice(scale_bboxes,[0,0,1],[-1,-1,1])
  # xmax = tf.add(xmin,tf.slice(scale_bboxes,[0,0,2],[-1,-1,1]))
  # ymax = tf.add(ymin,tf.slice(scale_bboxes,[0,0,3],[-1,-1,1]))
  # #tf bbox coor
  # tf_scale_bbox = tf.concat([ymin, xmin, ymax, xmax],2)

  tf_scale_bbox = bbox_tranformation_xywh_yxyx(scale_bboxes)
  image_with_bbox = tf.image.draw_bounding_boxes(images, tf_scale_bbox)
  tf.summary.image("resized_image_with_bbox"+name,image_with_bbox)

def process_image(encoded_image,
                  origin_bbox,
                  classes,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  image_format="jpeg"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image_summary("original_image", image)

  origin_size =tf.slice(tf.shape(image,"origin_size"),[0],[2])

  # Resize image.
#   assert (resize_height > 0) == (resize_width > 0)
#   if resize_height:
#     image = tf.image.resize_images(image,
#                                   size=[resize_height, resize_width],
#                                   method=tf.image.ResizeMethod.BILINEAR)

#   # Crop to final dimensions.
#   if is_training:
#     image = tf.random_crop(image, [height, width, 3])
#   else:
#     # Central crop, assuming resize_height > height, resize_width > width.
#     image = tf.image.resize_image_with_crop_or_pad(image, height, width)

# For now, do not random crop, otherwise bbox will shift
# resize bbox is needed as each picture are resized differently.
  image = tf.image.resize_images(image,
                                  size=[height, width],
                                  method=tf.image.ResizeMethod.BILINEAR)
  image_summary("resized_image", image)

  # start to cal resize bbox
  modify_size = tf.slice(tf.shape(image),[0],[2])

  # reverse image's shape as: image (n,h,w,c) bbox (x,y,w,h)
  origin_size = tf.reverse_v2(origin_size,[0])
  modify_size = tf.reverse_v2(modify_size,[0])

#   scale = tf.divide(modify_size, origin_size)
  # normalize bbox into 0 ~ 1
  scale = tf.divide(tf.ones([2], tf.float32), tf.cast(origin_size, tf.float32))
  scale = tf.expand_dims(tf.concat([scale,scale],0),0)
  scale_bbox = tf.multiply(tf.cast(origin_bbox,tf.float32),tf.cast(scale,tf.float32))

#   draw_tf_bounding_boxes(tf.expand_dims(image,0),tf.expand_dims(scale_bbox,0),name="target")

  # Randomly distort the image.
  if is_training:
    image = distort_image(image, thread_id)

  image_summary("final_image", image)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image, scale_bbox, classes

