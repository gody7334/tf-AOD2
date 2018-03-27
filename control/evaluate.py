from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import os.path
import time
import numpy as np

# get parsing argument
from utils.config import global_config
from graph.build_data import Build_Data
from graph.forward.lstm import LSTM
from graph.backward import Backward
from control.prepare_dataset import PrepareDataset

class Evaluate(object):
    
    def __init__(self):
        # inception inital func
        self.init_fn = None
        self.saver = None
        self.optimize = None
        self.g = None
        self.mode = 'train'
        self.prepared_dataset = None
        
        self.data = None
        self.model = None
        self.summary_op = None
        self.summary_writer = None
        self.global_step = None
        
    def run(self):
        # set evn to use CPU only
        global_config.assign_config()
        self.build_computation_graph()
        
        # export CUDA_VISIBLE_DEVICES=
        # export CUDA_VISIBLE_DEVICES=0
        # self.run_partial_graph()
        
    def build_computation_graph(self):
        # Build the TensorFlow graph.
        g = tf.Graph()
        with g.as_default():
            data = Build_Data(mode='eval')
            self.data = data
            
            model = Forward(mode='eval',data=data)
            self.model = model
            
            # self.setup_inception_initializer()
            
            # Create the Saver to restore model Variables.
            saver = tf.train.Saver()
            self.saver = saver
        
            # Create the summary operation and the summary writer.
            summary_op = tf.summary.merge_all()
            self.summary_op = summary_op
            
            summary_writer = tf.summary.FileWriter(global_config.global_config.eval_dir)
            self.summary_writer = summary_writer
        
            g.finalize()
            
            # sess run should under graph scope
            self.run_evaluation()
        
            # Run a new evaluation run every eval_interval_secs.
        self.g = g
        
    def run_evaluation(self):
        """Evaluates the latest model checkpoint.
        Args:
          model: Instance of ShowAndTellModel; the model to evaluate.
          saver: Instance of tf.train.Saver for restoring model Variables.
          summary_writer: Instance of FileWriter.
          summary_op: Op for generating model summaries.
        """
        # calculate performance index such as loss, AP(IoU)....
        def evaluate_model(sess):
            summary_str = sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.global_step)
            
            # Compute perplexity over the entire dataset.
            num_eval_batches = int(math.ceil(global_config.global_config.num_eval_examples / global_config.global_config.batch_size))
            
            start_time = time.time()
            sum_losses = 0.
            sum_weights = 0.
            for i in xrange(num_eval_batches):
                smooth_l1_losses, weights = sess.run([
                    self.model.target_smooth_l1_losses,
                    self.model.target_smooth_l1_losses_weights
                ])
                sum_losses += np.sum(smooth_l1_losses * weights)
                sum_weights += np.sum(weights)
                if not i % 100:
                    tf.logging.info("Computed losses for %d of %d batches.", i + 1, num_eval_batches)
            eval_time = time.time() - start_time
            
            perplexity = math.exp(sum_losses / sum_weights)
            tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)
            
            # Log perplexity to the FileWriter.
            summary = tf.Summary()
            value = summary.value.add()
            value.simple_value = perplexity
            value.tag = "Perplexity"
            self.summary_writer.add_summary(summary, self.global_step)
            
            # Write the Events file to the eval directory.
            self.summary_writer.flush()
            tf.logging.info("Finished processing evaluation at global step %d.", self.global_step)
                  
        # restore trained model
        def run_once():
            model_path = tf.train.latest_checkpoint(global_config.global_config.trained_model_checkpoint_dir)
            if not model_path:
                tf.logging.info("Skipping evaluation. No checkpoint found in: %s", global_config.global_config.trained_model_checkpoint_dir)
                return
            
            with tf.Session() as sess:
                # Load model from checkpoint.
                tf.logging.info("Loading model from checkpoint: %s", model_path)
                self.saver.restore(sess, model_path)
                self.global_step = tf.train.global_step(sess, self.model.global_step.name)
                tf.logging.info("Successfully loaded %s at global step = %d.", os.path.basename(model_path), self.global_step)
                
                if self.global_step < global_config.global_config.min_global_step:
                    tf.logging.info("Skipping evaluation. Global step = %d < %d", self.global_step, global_config.global_config.min_global_step)
                    return
            
                # Start the queue runners.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
            
                # Run evaluation on the latest checkpoint.
                try:
                    evaluate_model(sess=sess)
                except Exception, e:  # pylint: disable=broad-except
                    tf.logging.error("Evaluation failed.")
                    coord.request_stop(e)
            
                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)
               
        # evaluate model every eval_interval_secs 
        while True:
            start = time.time()
            tf.logging.info("Starting evaluation at " + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
            run_once()
            time_to_next_eval = start + global_config.global_config.eval_interval_secs - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)
            
    # def run_partial_graph(self):