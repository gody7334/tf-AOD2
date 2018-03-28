import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
import ipdb
from pprint import pprint as pp

from utils.config import global_config

class Solver(object):
    def __init__(self, model, optimizer, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 1000)
        self.batch_size = kwargs.pop('batch_size', global_config.global_config.batch_size)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 50)
        self.save_every = kwargs.pop('save_every', 50)
        self.log_path = kwargs.pop('log_path', global_config.global_config.tb_train_log_dir)
        self.model_path = kwargs.pop('model_path', global_config.global_config.tf_model_dir)
        self.pretrained_model = kwargs.pop('pretrained_model', global_config.global_config.tf_model_dir)
        self.test_model = kwargs.pop('test_model', global_config.global_config.tf_model_dir)
        self.mode = kwargs.pop('mode',global_config.global_config.mode)

        # set an optimizer by update rule
        # if self.update_rule == 'adam':
            # self.optimizer = tf.train.AdamOptimizer
        # elif self.update_rule == 'momentum':
            # self.optimizer = tf.train.MomentumOptimizer
        # elif self.update_rule == 'rmsprop':
            # self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self, chunk=0):
        # train/val dataset
        n_examples = self.data['bboxes'].shape[0]
        n_iters_per_epoch = int(np.floor(float(n_examples) / self.batch_size))
        features = self.data['features']
        images = self.data['images']
        bboxes = self.data['bboxes']
        classes = self.data['classes']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        val_iamges = self.val_data['images']
        n_iters_val = int(
            np.ceil(float(val_features.shape[0]) / self.batch_size))

        # build graphs for training model and sampling captions
        # loss = self.model.build_model()
        # print (loss)

        # train op
        # with tf.name_scope('optimizer'):
            # optimizer = self.optimizer(learning_rate=self.learning_rate)
            # grads = tf.gradients(loss, tf.trainable_variables())
            # grads_and_vars = list(zip(grads, tf.trainable_variables()))
            # train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # tf.get_variable_scope().reuse_variables()
        # _, _, generated_captions = self.model.build_sampler(max_len=20)

        # tODO generated bboxes for visualization when loc_sd = 0

        # summary op
        # tf.summary.scalar('batch_loss', loss)
        # for var in tf.trainable_variables():
            # tf.summary.histogram(var.op.name, var)

        # for grad, var in grads_and_vars:
            # tf.summary.histogram(var.op.name + '/gradient', grad)

        summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=10)

            if len(os.listdir(self.pretrained_model)):
                print(self.pretrained_model)
                if self.mode == 'train':
                    print "Start training with pretrained Model.."
                    saver.restore(sess, tf.train.latest_checkpoint(self.pretrained_model))

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            # for e in range(self.n_epochs):
            e = 0
            while(True):
                rand_idxs = np.random.permutation(n_examples)
                bboxes = bboxes[rand_idxs]
                classes = classes[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    classes_batch = classes[i * self.batch_size:(i + 1) * self.batch_size]
                    bboxes_batch = bboxes[i * self.batch_size:(i + 1) * self.batch_size]
                    image_idxs_batch = image_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    features_batch = features[image_idxs_batch]
                    images_batch = images[image_idxs_batch]
                    feed_dict = {self.model.images: images_batch,
                                 self.model.features: features_batch,
                                 self.model.bbox_seqs: bboxes_batch,
                                 self.model.class_seqs: classes_batch}
                    _, l = sess.run([self.optimizer.train_op, self.model.batch_loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(
                            summary, tf.train.global_step(sess, self.model.global_step))

                    # if (i + 1) % self.print_every == 0:
                        # print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1 + chunk * 10, i + 1, l)
                        # ground_truths = bboxes[image_idxs == image_idxs_batch[0]]
                        # decoded = decode_captions(
                            # ground_truths, self.model.idx_to_word)
                        # for j, gt in enumerate(decoded):
                            # print "Ground truth %d: %s" % (j + 1, gt)
                        # gen_caps = sess.run(generated_captions, feed_dict)
                        # decoded = decode_captions(
                            # gen_caps, self.model.idx_to_word)
                        # print "Generated caption: %s\n" % decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                # print out BLEU scores and file write
                # if self.print_bleu:
                    # all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    # for i in range(n_iters_val):
                        # features_batch = val_features[i *
                                                      # self.batch_size:(i + 1) * self.batch_size]
                        # feed_dict = {self.model.features: features_batch}
                        # gen_cap = sess.run(
                            # generated_captions, feed_dict=feed_dict)
                        # all_gen_cap[i *
                                    # self.batch_size:(i + 1) * self.batch_size] = gen_cap

                    # all_decoded = decode_captions(
                        # all_gen_cap, self.model.idx_to_word)
                    # save_pickle(
                        # all_decoded, "./data/val/val.candidate.captions.pkl")
                    # scores = evaluate(data_path='./data',
                                      # split='val', get_scores=True)
                    # write_bleu(scores=scores, path=self.model_path, epoch=e + chunk * 10)

                # save model's parameters
                e+=1
                if (e + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(
                        self.model_path, 'model'), global_step=e + 1 + chunk * 10)
                    print "model-%s saved." % (e + 1 + chunk * 10)

    def test(self, data, split='test', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(
            max_len=20)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(
                data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            # (N, max_len, L), (N, max_len)
            alps, bts, sam_cap = sess.run(
                [alphas, betas, sampled_captions], feed_dict)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" % decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t + 2)
                        plt.text(0, 1, '%s(%.2f)' % (
                            words[t], bts[n, t]), color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n, t, :].reshape(14, 14)
                        alp_img = skimage.transform.pyramid_expand(
                            alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(
                    np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i *
                                              self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch}
                    all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(
                        sampled_captions, feed_dict)
                all_decoded = decode_captions(
                    all_sam_cap, self.model.idx_to_word)
                save_pickle(
                    all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))
