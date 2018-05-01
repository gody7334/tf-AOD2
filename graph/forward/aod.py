from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.roi_pooling_layer import roi_pooling_op
from utils.roi_pooling_layer import roi_pooling_op_grad
import ipdb

from utils.config import global_config
from graph.ops import image_embedding
from graph.ops import image_processing
from graph.ops import inputs as input_ops
from graph.forward.Iforward import IForward
from graph.ops.debug import _debug_func

class AOD(IForward):
    def __init__(self, mode, data):
        IForward.__init__(self, mode, data)
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
                as its dimension is [N 14 14 512]
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        # TODO: move to global config
        # self.word_to_idx = word_to_idx
        # self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        # self._start = word_to_idx['<START>']
        # self._null = word_to_idx['<NULL>']
        # self.V = len(word_to_idx)

        self.prev2out = False
        self.ctx2out = True
        self.alpha_c = 0.0
        self.selector = True
        self.dropout = True
        # dimension of vggnet19 conv5_3 is [N 14 14 512]
        # self.L = 196
        # self.D = 512
        self.L = None # depend on desired inception layer
        self.D = None # depend on desired inception layer
        self.M = 512 # word embedded
        self.H = self.config.num_lstm_units # hidden stage
        self.T = self.config.num_time_step # Time step size of LSTM (how many predicting bboxes in a image)
        self.B = 4 # bbox dimension

        self.NN = self.config.batch_size # batch size
        self.WW = None # feature width
        self.HH = None # feature high
        self.DD = None # feature depth,(dimension)

        self.glimpse_w = 7 # glimpse width
        self.glimpse_h = 7 # glimpse high

        # Place holder for features and captions
        # self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        # self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.features = tf.placeholder(tf.float32, [None, 35, 35, 288])
        self.captions = None
        self.logits = None
        self.targets = None
        self.weights = None

        self.df = 0.5

        if self.mode == 'train':
            self.loc_sd = 0.1
            self.random_target_rate = 1.0
        else:
            self.loc_sd = 1e-10
            self.random_target_rate = 0.0

        self.ee_ratio = 0.9
        self.region_proposals_list = []
        self.roises_list = [] # RoI location, used to reconstuct the bbox coordinate in a image
        self.baselines_list = [] # policy baseline
        self.mean_locs_list = [] # expected location
        self.sampled_locs_list = [] # (agent) random sample from gaussion distribution
        self.sample_locs_origin_list = []
        self.bboxes_list = [] # predict bbox
        self.class_logits_list = [] # predict class
        self.rewards = None
        self.invalid_bbox_list = []

        self.region_proposals = None
        self.glimpses = None
        self.glimpses_project = None
        self.rois = None #
        self.roi_pooling_h = self.glimpse_h
        self.roi_pooling_w = self.glimpse_w

        # for debug
        self.bbox_input_removed_max = None
        self.predict_bbox_full = None
        self.iouses = None
        self.argmax_ious = None
        self.target_class = None
        self.target_bbox = None
        self.predict_bbox = None
        self.predict_class_logit = None
        self.rois = None
        self.class_losses = None
        self.bbox_losses = None
        self.policy_losses = None
        self.batch_loss = None

        # self._gen_first_region_proposal()
        self.build_image_embeddings()
        self.get_features_shape()
        self.build_model()
        self.get_loss()

    def build_seq_embeddings(self):
        return

    def build_image_embeddings(self):
        """ Load inception V3 graph (IForward) and post process the image feature map
        Inputs:
          self.images
        Outputs:
          self.image_embeddings
        """
        # call parent function to get original image features map
        # super(AOD, self).build_image_embeddings()

        # TOOD experiement different layer
        # inception_layer = self.inception_end_points['Mixed_5d']

        # get depth of image embedded
        # layer_shape = inception_layer.get_shape().as_list()
        # self.D = layer_shape[3]
        # self.L = layer_shape[1]*layer_shape[2]

        # RoI pooling need to retain W,H dim, no reshape needed
        # self.features = inception_layer
        return

    def get_features_shape(self):
        feature_shape = self.features.get_shape().as_list()
        # self.NN = feature_shape[0]
        self.WW = feature_shape[1]
        self.HH = feature_shape[2]
        self.DD = feature_shape[3]

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features = tf.reduce_mean(features,axis=3)
            features_flat = tf.reshape(features, [self.NN,-1])

            w_h = tf.get_variable('w_h', [self.WW*self.HH, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_flat, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.WW*self.HH, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_flat, w_c) + b_c)
            return c, h

    def _project_glimpses(self, glimpses, features, reuse=False):
        '''
        project rois dims to H-4 dims
        4 is for region proposal coordinate
        '''
        with tf.variable_scope('project_glimpses', reuse=reuse):
            # w = tf.get_variable('w', [self.glimpse_w*self.glimpse_h*self.DD+self.WW*self.HH, self.H-4],
                    # initializer=self.weight_initializer)
            # b = tf.get_variable('b', [self.H-4], initializer=self.const_initializer)
            # features = tf.reduce_mean(features,axis=3)
            # features_flat = tf.reshape(features, [self.NN,-1])
            # glimpses_flat = tf.reshape(glimpses, [self.NN, -1])
            # features_glimpses_flat = tf.concat([features_flat,glimpses_flat],1)
            # glimpses_proj = tf.nn.tanh(tf.matmul(features_glimpses_flat, w) + b)

            w = tf.get_variable('w', [self.glimpse_w*self.glimpse_h*self.DD, self.H-4],
                    initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.H-4], initializer=self.const_initializer)
            glimpses_flat = tf.reshape(glimpses, [self.NN, -1])
            glimpses_proj = tf.nn.tanh(tf.matmul(glimpses_flat, w) + b)

            return glimpses_proj

    def _attension_region_proposal_layer(self, h=None, reuse=False, t=0):
        # Here bbox def: (xmid,ymid,w,h) as easy to limit the boundry(0~1) and avoid revert coor
        with tf.variable_scope('attension_region_proposal_layer',reuse=reuse):
            baseline_w = tf.get_variable('baseline_w', [self.H, 1],initializer=self.weight_initializer)
            baseline_b = tf.get_variable('baseline_b', [1], initializer=self.const_initializer)
            mean_w = tf.get_variable('mean_w', [self.H, self.B],initializer=self.weight_initializer)
            mean_b = tf.get_variable('mean_b', [self.B],initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0))
            # mean_w = tf.get_variable('mean_w', [self.H, 2],initializer=self.weight_initializer)
            # mean_b = tf.get_variable('mean_b', [2],initializer=self.point5_initializer)

            # train a baseline_beline function
            # baseline might out of boundry.
            # h = tf.stop_gradient(h)
            baseline = tf.sigmoid(tf.matmul(h,baseline_w)+baseline_b)
            self.baselines_list.append(baseline)

            # compute next location
            # eye_center to the previous sample_loc:
            # high bias, bias to previous stage ,loc
            # low variance, limit to previous stage, loc
            eye_center = False
            if eye_center == False:
                mean_loc = tf.matmul(h,mean_w)+mean_b
            else:
                if t-1<0:
                    last_sample_loc = tf.cast(tf.convert_to_tensor(np.array([0.5,0.5,1.0,1.0])),tf.float32)
                else:
                    last_sample_loc = self.sampled_locs_list[t-1]
                mean_loc = tf.matmul(h,mean_w)+mean_b + last_sample_loc

            random_target_loc = tf.squeeze(tf.slice(tf.transpose(tf.random_shuffle(tf.transpose(self.bbox_seqs,(1,0,2))),(1,0,2)), [0,0,0],[-1,1,-1]))
            random_target_loc = self._convert_coordinate(random_target_loc, "mscoco", "frcnn",dim=2)
            random_target_loc.set_shape([self.NN, self.B])

            # when evaluation, remove random select target into sampling process
            random_mask = tf.random_uniform([self.NN, 1], 0, 1, tf.float32)
            random_mask = tf.cast(tf.less(random_mask, self.random_target_rate), tf.float32)

            mask = tf.cast(tf.equal(random_target_loc * random_mask, 0),tf.float32)
            # sample_loc_origin = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, self.loc_sd)
            sample_loc_origin = (mean_loc+tf.random_normal(mean_loc.get_shape(), 0, self.loc_sd))*(mask) + (random_target_loc+ tf.random_normal(mean_loc.get_shape(), 0, self.loc_sd*self.loc_sd))*(1-mask)

            self.mean_locs_list.append(mean_loc)
            self.sample_locs_origin_list.append(sample_loc_origin)

            sample_loc = sample_loc_origin
            # sample_loc = tf.stop_gradient(sample_loc)
            self.sampled_locs_list.append(sample_loc)

            return sample_loc

    def _ROI_pooling_layer(self, features, region_proposal, t):
        region_proposal = self._convert_coordinate(region_proposal, "frcnn", "bmp",dim=2)
        region_proposal =_debug_func(region_proposal,'roipooling_region_proposal_bmp',break_point=False, to_file=True)

        # convert from (0-1) to int coordinate
        xmin = tf.slice(region_proposal, [0,0],[-1,1])
        ymin = tf.slice(region_proposal, [0,1],[-1,1])
        xmax = tf.slice(region_proposal, [0,2],[-1,1])
        ymax = tf.slice(region_proposal, [0,3],[-1,1])

        n_idx = tf.expand_dims(tf.cast(tf.range(self.NN),tf.float32),1)
        xmin = tf.cast(tf.floor(xmin*(self.WW-1)),tf.float32) #(n,1)
        ymin = tf.cast(tf.floor(ymin*(self.HH-1)),tf.float32) #(n,1)
        xmax = tf.cast(tf.ceil(xmax*(self.WW-1)),tf.float32)  #(n,1)
        ymax = tf.cast(tf.ceil(ymax*(self.HH-1)),tf.float32)  #(n,1)
        rois = tf.concat([n_idx,xmin,ymin,xmax,ymax],1) #(n,5)

        # Q: pooling from 1x1 to 7x7? that all value is the same in 7x7 as region proposal is small?

        [y, argmax] = roi_pooling_op.roi_pool(features, rois, self.roi_pooling_w, self.roi_pooling_h, 1.0/3)

        # store rois for convert the coordinate between region <=> full image
        rois = tf.concat([xmin,ymin,xmax,ymax],1)
        rois =_debug_func(rois ,'roipooling_rois_bmp',break_point=False, to_file=True)
        rois = self._convert_coordinate(rois, "bmp", "frcnn",dim=2)
        rois =_debug_func(rois ,'roipooling_rois_frcnn',break_point=False, to_file=True)
        self.roises_list.append(rois)

        return y

    def _decode_lstm_bbox_class(self, h, reuse=False):
        '''
        @return
        bboxes (N,4) with faster-rcnn def (xmid, ymid, w, h)
        '''
        with tf.variable_scope('decode_bbox_class', reuse=reuse):

            w_fc6_bbox = tf.get_variable('w_fc6_bbox', [self.H, self.H*2], initializer=self.weight_initializer)
            b_fc6_bbox = tf.get_variable('b_fc6_bbox', [self.H*2], initializer=self.const_initializer)
            w_fc7_bbox = tf.get_variable('w_fc7_bbox', [self.H*2, self.H*2], initializer=self.weight_initializer)
            b_fc7_bbox = tf.get_variable('b_fc7_bbox', [self.H*2], initializer=self.const_initializer)

            w_fc6_class = tf.get_variable('w_fc6_class', [self.H, self.H*2], initializer=self.weight_initializer)
            b_fc6_class = tf.get_variable('b_fc6_class', [self.H*2], initializer=self.const_initializer)
            w_fc7_class = tf.get_variable('w_fc7_class', [self.H*2, self.H*2], initializer=self.weight_initializer)
            b_fc7_class = tf.get_variable('b_fc7_class', [self.H*2], initializer=self.const_initializer)

            w_bbox = tf.get_variable('w_bbox', [self.H*2,(self.config.num_classes)*4], initializer=self.weight_initializer)
            b_bbox = tf.get_variable('b_bbox', [(self.config.num_classes)*4], initializer=self.point5_initializer)

            w_class = tf.get_variable('w_class', [self.H*2, self.config.num_classes], initializer=self.weight_initializer)
            b_class = tf.get_variable('b_class', [self.config.num_classes], initializer=self.const_initializer)

            fc6_bbox = tf.nn.tanh(tf.matmul(h, w_fc6_bbox) + b_fc6_bbox)
            fc7_bbox = tf.nn.tanh(tf.matmul(fc6_bbox, w_fc7_bbox) + b_fc7_bbox)

            fc6_class = tf.nn.tanh(tf.matmul(h, w_fc6_class) + b_fc6_class)
            fc7_class = tf.nn.tanh(tf.matmul(fc6_class, w_fc7_class) + b_fc7_class)

            # here only unnormalize log probabilities (logits, score) for each class, need softmax & argmax to find the class
            class_logits = tf.matmul(fc7_class, w_class) + b_class

            # do not clip the boundary, let optimisation to do the job
            bboxes = tf.matmul(fc7_bbox, w_bbox) + b_bbox
            bboxes = tf.reshape(bboxes,[self.NN, self.config.num_classes,4])
            self.invalid_bbox_list.append(self._invalid_bbox(bboxes))

            # get predict bbox with max class prediction
            max_class_index = tf.expand_dims(tf.argmax(class_logits,axis=1),1)
            n_idx = tf.cast(tf.expand_dims(tf.range(self.NN),1),tf.int64)
            n_idx = tf.concat([n_idx,max_class_index],1)
            bboxes = tf.gather_nd(bboxes, n_idx)

            self.bboxes_list.append(bboxes)
            self.class_logits_list.append(class_logits)

            return (bboxes, class_logits)

    def _convert_bbox_to_full_image_coordinate(self,roises,bboxes):
        '''
        convert predict bbox from regional coordinate back to full image coordinate
        using region of interest(used for rois pooling)

        @param
        roises (N,4): region of insterests, frcnn format with
            boundary HH and WW, used for rois pooling
        bboxes (N,4): regional predict bbox, frcnn format with
            boundary 0~1

        @return
        bbox_coor_full (N, 4): full image bbox prediction with frcnn format

        '''
        # Todo convert logit(bbox) result from region to full image coordinate
        roises =_debug_func(roises ,'fullcoor_roises',break_point=False, to_file=True)
        bboxes =_debug_func( bboxes,'fullcoor_bboxes_regional',break_point=False, to_file=True)
        # roises = tf.stop_gradient(roises)
        rp_xmid = tf.slice(roises, [0,0],[-1,1])
        rp_ymid = tf.slice(roises, [0,1],[-1,1])
        rp_w = tf.slice(roises, [0,2],[-1,1])
        rp_h = tf.slice(roises, [0,3],[-1,1])

        # convert from WW, HH to 0~1
        rp_xmid = rp_xmid/(self.WW-1)
        rp_ymid = rp_ymid/(self.HH-1)
        rp_w = rp_w/(self.WW-1)
        rp_h = rp_h/(self.HH-1)

        bbox_xmid = tf.slice(bboxes, [0,0],[-1,1])
        bbox_ymid = tf.slice(bboxes, [0,1],[-1,1])
        bbox_w = tf.slice(bboxes, [0,2],[-1,1])
        bbox_h = tf.slice(bboxes, [0,3],[-1,1])

        bbox_xmid_full = rp_xmid + rp_w*bbox_xmid - (rp_w/2.0)
        bbox_ymid_full = rp_ymid + rp_h*bbox_ymid - (rp_h/2.0)
        bbox_w_full = rp_w*bbox_w
        bbox_h_full = rp_h*bbox_h

        bbox_coor_full = tf.concat([bbox_xmid_full,bbox_ymid_full,bbox_w_full,bbox_h_full],1)
        bbox_coor_full =_debug_func(bbox_coor_full,'fullcoor_bbox_coor_full',break_point=False, to_file=True)

        return bbox_coor_full

    def get_argmax_ious_class_bbox(self, iou_input, class_input, bbox_input):
        '''
        Prepare target in each episode
        ps: prepare target should stop gradint
        will compute iou loss later

        @ param
        iou_input (N,T) all iou between predict and target bbox pair
        class_input (N,T) all classes in images
        bbox_input (N,T,4) all bboxes in images

        @ return
        argmax_ious (N) max ious index (within T) (-1 if ious < 0.5, represent background)
        class (N) max ious class (insert background class)
        bbox (N,4) max ious bbox (0,0,0,0 for background bbox)
        '''

        # filter iou < 0.5
        # mask = tf.stop_gradient(tf.greater(iou_input,0.5))
        # ipdb.set_trace()
        mask = tf.greater(iou_input,0.5)
        ious = tf.multiply(iou_input,tf.cast(mask, tf.float32))

        # select max and its index
        max_ious = tf.reduce_max(ious,1)
        argmax_ious = tf.argmax(ious,1)

        n_idx = tf.expand_dims(tf.range(self.NN),1)
        max_ious_index = tf.concat([tf.cast(n_idx,tf.int64),argmax_ious],1) #(n,2)

        empty_bbox = tf.SparseTensor(max_ious_index, tf.fill([self.NN],-1.0), tf.slice(tf.shape(bbox_input,out_type=tf.int64),[0],[2]))
        bbox_untarget_mask = tf.fill(tf.shape(bbox_input),1.0)+tf.expand_dims(tf.sparse_tensor_to_dense(empty_bbox),[2])
        self.bbox_input_removed_max = bbox_input*bbox_untarget_mask


        # filter max_ious = 0, let index = num_class+1 as Background
        # obj_mask = tf.stop_gradient(tf.cast(tf.not_equal(max_ious,0.0), tf.int64))
        obj_mask = tf.cast(tf.not_equal(max_ious,0.0), tf.int64)
        # back_mask = tf.stop_gradient(tf.cast(tf.equal(max_ious,0.0), tf.int64))
        back_mask = tf.cast(tf.equal(max_ious,0.0), tf.int64)

        # get max_iou index
        N = self.NN
        background_class = self.config.num_classes-1
        n_idx = tf.expand_dims(tf.cast(tf.range(N),tf.int64),1)
        # argmax_ious = tf.expand_dims(argmax_ious,1)
        n_idx = tf.concat([n_idx,argmax_ious],1)

        # get max_iou class, bbox
        classes = tf.gather_nd(class_input, n_idx)*tf.squeeze(obj_mask) + tf.squeeze(back_mask)*background_class
        bboxes = tf.gather_nd(bbox_input, n_idx)*tf.cast(tf.expand_dims(tf.squeeze(obj_mask),1),tf.float32)
        argmax_ious = tf.squeeze(argmax_ious)

        return (argmax_ious, classes, bboxes)

    def get_class_softmax_loss(self, logit_input, label_input):
        '''
        tf.nn.sparse_softmax_cross_entropy_with_logits
        will apply: softmax to logit_input, one_hot encoded to label_input
        then compute cross-entropy between (above) two value to get losses

        @ param
        logit_input (N,num_class) contains computed scores (unnormalized log probabilities) for each class
        label_input (N) True class, values are within [0,num_class-1]

        @ return
        losses (N), batches of softmax loss between logit and label
        '''
        # mask = tf.stop_gradient(tf.less_equal(label_input,91))
        # Down sample background class by random filtering out background loss
        bg_class_mask = tf.cast(tf.equal(label_input,self.config.num_classes-1),tf.int64)
        obj_class_mask = tf.cast(tf.not_equal(label_input, self.config.num_classes-1), tf.float32)

        down_sample_mask = tf.random_uniform(bg_class_mask.get_shape(), minval=0, maxval=100, dtype=tf.int64)
        down_sample_mask = tf.cast(tf.greater(down_sample_mask, 98),tf.int64)
        bg_class_mask = tf.cast(bg_class_mask * down_sample_mask, tf.float32)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_input, logits=logit_input)
        losses = losses * bg_class_mask + losses * obj_class_mask
        return losses

    def get_bbox_iou_loss(self, predict_bbox_input, target_bbox_input):
        '''
        compute iou then -log(iou) to get log losses scale
        as iou within 0~1 the -log loss within infinte ~ 0
        ps: deal with infinte value by add small value to iou

        @ param
        predict_bbox_input (N,4) with faster-rcnn def (xmid, ymid, w, h)
        target_bbox_input (N,4) with mscoco def (xmin, ymin, w, h)

        @ return
        iou_losses (N), batches of iou loss between predict and target bbox
        ps: after add 1e-10, losses are within 23~0
        however, as filter out iou < 0.5 to let it as background(bbox = (0,0,0,0) ), it will cause
        uncontinuous losses, might have potential problem,
        on the other hand, if we don't count background bbox losses,
        model will prefer get background bbox rather than object bbox, which is not we want.
        '''
        ious = self.get_iou(tf.expand_dims(predict_bbox_input,1),tf.expand_dims(target_bbox_input,1))
        ious = tf.squeeze(ious,[1,2])
        iou_losses = -1.0 * tf.log(tf.clip_by_value(ious,1e-10,1))
        mask = tf.greater(ious,0)
        iou_losses = tf.multiply(iou_losses,tf.cast(mask, tf.float32))

        return iou_losses

    def get_policy_gradient_loss(self, num_class_input,
                                target_class_input,
                                target_bbox_input,
                                predict_class_input,
                                predict_bbox_input,
                                mean_location_input,
                                sample_location_input,
                                sample_location_origin,
                                baseline_input):
        '''
        compute policy reward and function approximator (loss) in each time step

        @ param
        num_class (int) number of class (including background) for one hot encoding
        target_class_input (N, T) the target class label, which selected using argmax_iou in each time step
        target_bbox_input (N, T, 4) the target bbox, which selected using argmax_iou in each time step, mscoco fomat
        predict_class_input (N, T, num_class) the predict class, unnormalized logit, in each time step
        predict_bbox_input (N, T, 4) the predict bbox, in each time step, frcnn format
        mean_location_input (N, T, 4)
        sample_location_input (N, T, 4)

        @ return
        policy_gradient_loss (N), batches of iou loss between predict and target bbox
        '''

        # to use for maximum likelihood with input location
        def gaussian_pdf(mean, sample):
            Z = 1.0 / (self.loc_sd * tf.sqrt(2.0 * np.pi))
            a = -tf.square(sample - mean) / (2.0 * tf.square(self.loc_sd))
            return Z * tf.exp(a)

        def cum_discount_rewards(rewards, num_step, df=0.99):
            from scipy.ndimage.interpolation import shift
            cum_prod_reward_list = []
            for s in range(num_step):
                dfs = np.zeros(num_step)
                dfs[:] = df
                dfs[0] = 1.0
                dfs = np.cumprod(dfs)
                dfs = shift(dfs, s, cval=0.0)
                dfs = tf.convert_to_tensor(dfs, dtype=tf.float32)
                cum_prod_reward = tf.reduce_sum(rewards*dfs, axis=1)
                cum_prod_reward_list.append(cum_prod_reward)
            return tf.transpose(tf.stack(cum_prod_reward_list),(1,0))

        predict_class_input = tf.stop_gradient(predict_class_input)
        predict_bbox_input = tf.stop_gradient(predict_bbox_input)

        target_class_one_hot = tf.one_hot(target_class_input, num_class_input)
        predict_class_prob = tf.nn.softmax(predict_class_input)
        predict_class_prob =_debug_func(predict_class_prob ,'policy_predict_class_prob',break_point=False, to_file=True)
        target_class_one_hot =_debug_func(target_class_one_hot,'policy_target_class_one_hot',break_point=False, to_file=True)
        predict_target_prob = tf.reduce_sum(predict_class_prob*target_class_one_hot, axis=2)
        predict_target_prob =_debug_func(predict_target_prob ,'policy_predict_target_prob',break_point=False, to_file=True)

        target_bbox_input =_debug_func(target_bbox_input ,'policy_target_bbox_input',break_point=False, to_file=True)
        predict_bbox_input =_debug_func(predict_bbox_input ,'policy_predict_bbox_input',break_point=False, to_file=True)

        iou = self.get_iou(predict_bbox_input,target_bbox_input,"_policy")
        baseline_iou = tf.squeeze(baseline_input,[2])
        baseline_iou_stop_gradient = tf.stop_gradient(baseline_iou)
        # baseline_iou = tf.reduce_mean(tf.fill(iou.get_shape(), 0.2),[2])
        iou =_debug_func(tf.squeeze(iou,[2]) ,'policy_iou',break_point=False, to_file=True)
        baseline_iou =_debug_func(baseline_iou ,'policy_baseline_iou',break_point=False, to_file=True)

        invalid_mean_loc = self._invalid_bbox(mean_location_input)
        invalid_sample_loc = self._invalid_bbox(sample_location_origin)
        invalid_sample_area = self._invalid_area(sample_location_origin)
        invalid_mean_loc =_debug_func(invalid_mean_loc ,'policy_invalid_mean_loc',break_point=False, to_file=True)
        invalid_sample_loc =_debug_func(invalid_sample_loc ,'policy_invalid_sample_loc',break_point=False, to_file=True)

        # rewards
        rewards_scale = 1e1
        invalid_scale = 1e0
        invalid_area_scale = 1e0
        # rewards = (tf.squeeze(iou,[1,2]))*rewards_scale
        # rewards = (tf.squeeze(iou,[1,2]) * predict_target_prob)*rewards_scale
        # rewards = (tf.reduce_sum(iou,[2]))*rewards_scale - invalid_scale*(tf.reduce_mean((invalid_sample_loc),[2]) + tf.reduce_mean((invalid_mean_loc),[2]))
        rewards = rewards_scale * iou * predict_target_prob
        rewards =_debug_func(rewards ,'policy_rewards_step',break_point=False, to_file=True)
        # cum_rewards = tf.cumsum(rewards,axis=1,reverse=True)
        cum_rewards = cum_discount_rewards(rewards, self.T , df=self.df)
        cum_rewards = cum_rewards - invalid_scale*(tf.reduce_mean((invalid_sample_loc),[2])) - invalid_sample_area*invalid_area_scale
        # cum_rewards = cum_rewards - invalid_scale*(tf.reduce_mean((invalid_sample_loc),[2]) + tf.reduce_mean((invalid_mean_loc),[2]))
        cum_rewards =_debug_func(cum_rewards,'policy_cum_rewards',break_point=False, to_file=True)

        self.rewards = rewards

        mean_location_input =_debug_func(mean_location_input,'policy_mean_location_input',break_point=False, to_file=True)
        sample_location_input =_debug_func(sample_location_input ,'policy_sample_location_input',break_point=False, to_file=True)
        sample_location_origin =_debug_func(sample_location_origin ,'policy_sample_location_origin',break_point=False, to_file=True)

        # construct schocastic policy using mean and sample location
        # p_loc = gaussian_pdf(mean_location_input, sample_location_input)
        p_loc = gaussian_pdf(mean_location_input, sample_location_origin)
        # p_loc = (-0.5)*((mean_location_input - sample_location_origin)**2)
        p_loc =_debug_func(p_loc ,'policy_p_loc',break_point=False, to_file=True)

        # likelihood estimator
        J = tf.reduce_sum(tf.reduce_sum(tf.log(p_loc + 1e-10),[2]) * (cum_rewards - baseline_iou_stop_gradient),[1])
        # J = tf.reduce_sum(tf.reduce_sum(p_loc + 1e-10,[2]) * (cum_rewards - baseline_iou_stop_gradient),[1])
        J = J - tf.reduce_sum(tf.square(cum_rewards - baseline_iou), 1)
        J =_debug_func(J ,'policy_J',break_point=False, to_file=True)
        return -J

    def build_model(self):
        features = self.features    #(N,W,H,D)

        # batch normalize feature vectors
        with tf.variable_scope('batch_norm'):
            features = self._batch_norm(features, mode='train', name='features')

        batch_size = self.NN

        # initial lstm c,h with features
        (c,h) = self._get_initial_lstm(features)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)


        for t in range(self.T):
            region_proposals = self._attension_region_proposal_layer(h,reuse=(t!=0),t=t)
            self.region_proposals_list.append(region_proposals)

            self.glimpses = self._ROI_pooling_layer(features, region_proposals, t)

            self.glimpses_project = self._project_glimpses(self.glimpses,features,reuse=(t!=0))
            # with tf.variable_scope('batch_norm', reuse=(t != 0)):
                # self.glimpses_project = self._batch_norm(self.glimpses_project, mode='train', name='glimpses_project')
            self.glimpses_project = tf.concat([self.glimpses_project,tf.cast(region_proposals,tf.float32)],1) #(N,H)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=self.glimpses_project, state=[c, h])

            h =_debug_func(h ,'buildmodel_h',break_point=False, to_file=True)
            self._decode_lstm_bbox_class(self.glimpses_project, reuse=(t!=0))

        # self.region_proposals = tf.transpose(tf.stack(region_proposals_list),(1,0))

    def get_loss(self):
        # self.roises_list = [] # RoI location, used to reconstuct the bbox coordinate in a image
        # self.baselines_list = [] # policy baseline
        # self.mean_locs_list = [] # expected location
        # self.sampled_locs_list = [] # (agent) random sample from gaussion distribution
        # self.bboxes_list = [] # predict bbox
        # self.class_logits_list = [] # predict class

        class_losses_list = []
        bbox_losses_list = []
        target_classes_list = []
        target_bboxes_list = []
        policy_losses_list = []
        predict_bbox_full_list = []
        self.bbox_input_removed_max = tf.identity(self.bbox_seqs)

        for t in range(self.T):
            predict_bbox = self.bboxes_list[t]
            predict_class_logit = self.class_logits_list[t]
            rois = self.roises_list[t]
            sample_location = tf.cast(self.sampled_locs_list[t], tf.float32)
            # rois = tf.stop_gradient(rois)

            predict_bbox_full = self._convert_bbox_to_full_image_coordinate(rois, predict_bbox)
            predict_bbox_full_list.append(predict_bbox_full)
            predict_bbox_full =_debug_func(predict_bbox_full ,'getloss_predict_bbox_full',break_point=False, to_file=True)

            # iouses = self.get_iou(tf.expand_dims(predict_bbox_full,1), self.bbox_seqs)
            sample_location =_debug_func(sample_location ,'getloss_sample_location',break_point=False, to_file=True)
            # self.bbox_seqs =_debug_func(self.bbox_seqs ,'getloss_self.bbox_seqs',break_point=False, to_file=True)

            iouses = self.get_iou(tf.expand_dims(sample_location,1), self.bbox_seqs, mode="")
            iouses =_debug_func(iouses ,'getloss_iousess',break_point=False, to_file=True)

            argmax_ious, target_class, target_bbox = self.get_argmax_ious_class_bbox(iouses, self.class_seqs, self.bbox_input_removed_max)
            target_classes_list.append(target_class)
            target_bboxes_list.append(target_bbox)
            argmax_ious =_debug_func(argmax_ious ,'getloss_argmax_ious',break_point=False, to_file=True)
            target_class =_debug_func(target_class ,'getloss_target_class',break_point=False, to_file=True)
            target_bbox =_debug_func(target_bbox ,'getloss_target_bbox',break_point=False, to_file=True)

            class_losses = self.get_class_softmax_loss(predict_class_logit, target_class)

            mask = tf.cast(tf.greater(tf.reduce_sum(target_bbox,[1]),0),tf.float32)
            # bbox_losses = self.get_bbox_iou_loss(predict_bbox_full, target_bbox)
            # bbox_losses = (tf.reduce_sum(invalid_bbox*0.01,[1]) + bbox_losses)*mask
            bbox_losses = self._smooth_l1_loss(self._convert_coordinate(predict_bbox_full, "frcnn","mscoco",dim=2),target_bbox)
            bbox_losses = tf.reduce_sum(bbox_losses, [1])*mask
            # bbox_losses = tf.reduce_sum(self._smooth_l1_loss(
                # self._convert_coordinate(predict_bbox_full, "frcnn","mscoco",dim=2)
                # ,target_bbox),[1])*mask

            class_losses_list.append(class_losses)
            bbox_losses_list.append(bbox_losses)

        target_classes = tf.transpose(tf.stack(target_classes_list),(1,0))
        target_bboxes = tf.transpose(tf.stack(target_bboxes_list),(1,0,2))
        predict_class_logits = tf.transpose(tf.stack(self.class_logits_list),(1,0,2))
        predict_bbox_full = tf.transpose(tf.stack(predict_bbox_full_list),(1,0,2))
        mean_location = tf.transpose(tf.stack(self.mean_locs_list),(1,0,2))
        sample_location = tf.transpose(tf.stack(self.sampled_locs_list),(1,0,2))
        sample_location_origin = tf.transpose(tf.stack(self.sample_locs_origin_list),(1,0,2))
        baseline = tf.transpose(tf.stack(self.baselines_list),(1,0,2))

        policy_losses = self.get_policy_gradient_loss(self.config.num_classes,
                            target_classes,
                            target_bboxes,
                            predict_class_logits,
                            predict_bbox_full,
                            mean_location,
                            sample_location,
                            sample_location_origin,
                            baseline)

        class_losses = tf.transpose(tf.stack(class_losses_list),(1,0))
        bbox_losses = tf.transpose(tf.stack(bbox_losses_list),(1,0))
        total_invalid_bbox = tf.transpose(tf.stack(self.invalid_bbox_list),(1,0,2,3))


        class_loss = tf.reduce_mean(class_losses)
        bbox_loss = tf.reduce_mean(bbox_losses)
        policy_loss = tf.reduce_mean(policy_losses)
        total_invalid_bbox = tf.reduce_mean(tf.reduce_sum(total_invalid_bbox,[1,2,3]))
        reward = tf.reduce_sum(self.rewards)

        batch_loss = class_loss*1.0 + bbox_loss*1.0 + policy_loss*1.0 + total_invalid_bbox*0.1

        batch_loss = self._l2_regularization(batch_loss)

        self.predict_bbox = predict_bbox
        self.predict_class_logit = predict_class_logit
        self.rois = rois
        self.predict_bbox_full = predict_bbox_full
        self.iouses = iouses
        self.argmax_ious = argmax_ious
        self.target_class = target_class
        self.target_bbox = target_bbox
        self.class_losses = class_loss
        self.bbox_losses = bbox_loss
        self.policy_losses = policy_loss
        self.batch_loss = batch_loss

        logits = predict_bbox_full
        self.logits = logits
        logits =_debug_func(logits ,'getloss_logits',break_point=False, to_file=True)

        logits = self._convert_coordinate(logits, "frcnn","mscoco",dim=3)
        image_processing.draw_tf_bounding_boxes(self.images, logits, name="logits")
        image_processing.draw_tf_bounding_boxes(self.images, self.bbox_seqs, name="targets")

        target_mask = tf.cast(tf.greater(target_bboxes,0),tf.float32)
        predict_bbox_full_on_target = target_mask*predict_bbox_full
        region_proposal = tf.transpose(tf.stack(self.region_proposals_list),(1,0,2))
        region_proposal_on_target = target_mask*region_proposal
        region_proposal = self._convert_coordinate(region_proposal, "frcnn","mscoco",dim=3)
        predict_bbox_full_on_target = self._convert_coordinate(predict_bbox_full_on_target, "frcnn","mscoco",dim=3)
        region_proposal_on_target = self._convert_coordinate(region_proposal_on_target , "frcnn","mscoco",dim=3)
        image_processing.draw_tf_bounding_boxes(self.images, region_proposal, name="region_proposal")
        image_processing.draw_tf_bounding_boxes(self.images,predict_bbox_full_on_target, name="predict_bbox_full_on_target")
        image_processing.draw_tf_bounding_boxes(self.images,region_proposal_on_target, name="region_proposal_on_target")

        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        # Add summaries.
        tf.summary.scalar("losses/reward", reward)
        tf.summary.scalar("losses/class_loss", class_loss)
        tf.summary.scalar("losses/bbox_loss", bbox_loss)
        tf.summary.scalar("losses/policy_loss", policy_loss)
        tf.summary.scalar("losses/batch_loss", batch_loss)
        tf.summary.scalar("losses/total_loss", total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

        self.total_loss = total_loss
        self.target_smooth_l1_losses = batch_loss  # Used in evaluation.
        # self.target_smooth_l1_losses_weights = weights  # Used in evaluation.

        return total_loss


