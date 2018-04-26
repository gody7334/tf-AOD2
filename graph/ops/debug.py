# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint as pp
import pprint
from time import sleep
import random
import pdb
from utils.config import global_config

counter = 0

def _debug_func(tensor, name="", print_tf=False, print_op=False, break_point=False, to_file=False):
  gc = global_config.global_config
  global counter
  counter = gc.train_step_count

  if gc.is_tf_log:
    t = tensor
    debug_op = tf.py_func(debug_func, [t, t.name, str(t.op), str(t.dtype), t.device, print_tf, print_op, break_point, name, to_file], [tf.bool])

    with tf.control_dependencies(debug_op):
      tensor = tf.identity(tensor, name=name)
  else:
    tensor = tensor

  return tensor

def debug_func(tf, tf_name, tf_op, tf_type, tf_device, print_tf, print_op, break_point, name, to_file):
  global counter
  if to_file and counter%50==0:
    np.set_printoptions(threshold=np.nan)
    f = open(global_config.global_config.tf_log_dir + name + '_log.txt','a')
    pprint.pprint(name, f)
    pprint.pprint(tf_name, f)
    pprint.pprint(np.round(tf,2), f)
    # f.write('name: '+pprint.pformat(name))
    # f.write('tf_name: '+pprint.pformat(tf_name))
    # f.write('tf_shape: '+pprint.pformat(tf.shape))
    # f.write('tf_type: '+pprint.pformat(tf_type))
    # f.write('tf_device: {} '.format(tf_device))
    # f.write('tf_element: {} '.format(tf))
    # f.write('tf_op: {} '.format(tf_op))
    # f.write('')
    f.close()
  elif to_file == False:
    sleep((random.randint(0, 10) / 1000))
    print('name: {}'.format(name))
    print('tf_name: {}'.format(tf_name))
    print('tf_shape: {} '.format(tf.shape))
    print('tf_type: {} '.format(tf_type))
    print('tf_device: {} '.format(tf_device))
    print('')

    if print_tf:
      np.set_printoptions(threshold=50)
      print('tf_element: ')
      pp(tf)
    if print_op:
      print('tf_op: ')
      print(tf_op)
    if break_point:
      # name, tf_name, tf_shape, tf_type, tf_device, tf_op, tf
      pdb.set_trace()  # BREAKPOINT

  return False
