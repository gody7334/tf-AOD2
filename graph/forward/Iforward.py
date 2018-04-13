from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import ipdb

from utils.config import global_config
from graph.ops import image_embedding
from graph.ops import image_processing
from graph.ops import inputs as input_ops
from graph.ops.debug import _debug_func

class IForward(object):

    def __init__(self, mode, data):
        assert mode in ["train", "val", "inference"], "mode should be 'train', 'val', or 'inference"
        self.config = global_config.global_config
        self.mode = mode
        self.data = data

        self.train_inception = global_config.parse_args.train_inception

        # if data is not None:
        if data is None:
            # A float32 Tensor with shape [batch_size, height, width, channels].
            # self.images = data.images
            self.images = tf.placeholder(tf.float32, [None, 299, 299, 3])

            # An int32 Tensor with shape [batch_size, padded_length].
            # self.input_seqs = data.input_seqs

            # An int32 Tensor with shape [batch_size, padded_length].
            # self.target_seqs = data.target_seqs

            # An int32 0/1 Tensor with shape [batch_size, padded_length].
            # self.input_mask = data.input_mask

            # An int32 Tensor with shape [batch_size, padded_length].
            # self.bbox_seqs = data.bbox_seqs
            self.bbox_seqs = tf.placeholder(tf.float32, [None, None, 4])

            # An int32 Tensor with shape [batch_size, padded_length].
            # self.class_seqs = data.class_seqs
            self.class_seqs = tf.placeholder(tf.int64, [None, None])

            # An int32 0/1 Tensor with shape [batch_size, padded_length].
            # self.bbox_mask = data.bbox_mask

            # An int32 0/1 Tensor with shape [batch_size, padded_length].
            # self.class_mask = data.class_mask

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_smooth_l1_losses = None  # Used in evaluation.

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_smooth_l1_losses_weights = None  # Used in evaluation.

        # inception model's graph
        self.inception_output = None
        self.inception_end_points = None

        # initializer
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)
        self.random_normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.point5_initializer = tf.constant_initializer(0.5)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # A global variable for training graph
        self.global_step = None

        self.setup_global_step()
        self.build_seq_embeddings()

        self.center = None
        self.center_l1_losses = None
        self.invalid_totla_loss = None

    def build_seq_embeddings(self):
        return

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.
           pass image into inceptionV3 and get image features map (add full connected layer at the end)
        Inputs:
          self.images

        Outputs:
          self.image_embeddings
        """
        inception_output, inception_end_points = image_embedding.inception_v3(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        self.inception_output = inception_output
        self.inception_end_points = inception_end_points

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

    def _convert_coordinate(self, x, type_from, type_to, dim=3):
        '''
        type:
        bmp:(xmin, ymin, xmax, ymax)
        tf:(ymin, xmin, ymax, xmax)
        mscoco:(xmin, ymin, w, h)
        frcnn:(xmid, ymid, w, h)
        '''
        if dim == 2:
            x = tf.expand_dims(x,1)

        x1 = tf.slice(x, [0,0,0],[-1,-1,1])
        x2 = tf.slice(x, [0,0,1],[-1,-1,1])
        x3 = tf.slice(x, [0,0,2],[-1,-1,1])
        x4 = tf.slice(x, [0,0,3],[-1,-1,1])

        if type_from == "frcnn" and type_to == "bmp":
            x1 = x1-x3/2.0
            x2 = x2-x4/2.0
            x3 = x1+x3
            x4 = x2+x4
            x = tf.concat([x1,x2,x3,x4],-1)

        if type_from == "bmp" and type_to == "frcnn":
            x3 = x3-x1
            x4 = x4-x2
            x1 = x1 + x3/2.0
            x2 = x2 + x4/2.0
            x = tf.concat([x1,x2,x3,x4],-1)

        if type_from == "mscoco" and type_to == "bmp":
            x3 = x1+x3
            x4 = x2+x4
            x = tf.concat([x1,x2,x3,x4],-1)

        if type_from == "bmp" and type_to == "mscoco":
            x3 = x3-x1
            x4 = x4-x2
            x = tf.concat([x1,x2,x3,x4],-1)

        if type_from == "frcnn" and type_to == "mscoco":
            x1 = x1-x3/2.0
            x2 = x2-x4/2.0
            x = tf.concat([x1,x2,x3,x4],-1)

        if type_from == "mscoco" and type_to == "frcnn":
            x1 = x1+x3/2.0
            x2 = x2+x4/2.0
            x = tf.concat([x1,x2,x3,x4],-1)

        if dim == 2:
            x = tf.squeeze(x,[1])

        return x

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def _smooth_l1_loss(
            self,
            bbox_pred,
            bbox_targets,
            bbox_inside_weights=1.0,
            bbox_outside_weights=1.0,
            sigma=1.0,
            dim=[1]):

        bbox_pred =_debug_func(bbox_pred,'l1loss_bbox_pred',break_point=False, to_file=True)
        bbox_targets=_debug_func(bbox_targets,'l1loss_bbox_targets',break_point=False, to_file=True)

        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        # stop_gradient used to not compute gradient in back prap,
        # as its only used to compute smoothL1 sign, not in the computation graph for optimization
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box

        # loss_box = tf.reduce_mean(tf.reduce_sum(
        #   out_loss_box,
        #   axis=dim
        # ))
        # return loss_box

        # donnot sum loss yet
        return out_loss_box

    def _get_min_smooth_l1_loss(
            self,
            bbox_pred,
            bbox_targets,
            bbox_inside_weights=1.0,
            bbox_outside_weights=1.0,
            sigma=1.0,
            dim=[1]):
        """
        get the min loss from bboxes
        as there is no real order between bboxes,
        The target bbox for a predicted bbox can be any bbox within the bboxes
        bbox_pred (N,Lt,4) -> (N,Lt,L,4)
        bbox_targets(N,L,4) -> (N,Lt,L,4) -> (N,Lt,L,4)
        """
        bbox_pred_bc = tf.expand_dims(bbox_pred,2)
        bbox_targets_bc = tf.expand_dims(bbox_targets,1)

        sigma_2 = sigma ** 2
        box_diff_bc = bbox_pred_bc - bbox_targets_bc
        in_box_diff_bc = bbox_inside_weights * box_diff_bc
        abs_in_box_diff_bc = tf.abs(in_box_diff_bc)
        abs_in_box_diff = tf.reduce_min(abs_in_box_diff_bc, 2)
        # stop_gradient used to not compute gradient in back prap,
        # as its only used to compute smoothL1 sign, not in the computation graph for optimization
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(abs_in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box

        # loss_box = tf.reduce_mean(tf.reduce_sum(
        #   out_loss_box,
        #   axis=dim
        # ))
        # return loss_box

        # donnot sum loss yet
        return out_loss_box

    def _l2_regularization(self, loss, name = "l2_regularization"):
        with tf.name_scope(name):
            lambda_l2_reg = 1e-5
            l2 = sum(
                tf.nn.l2_loss(tf_var)*lambda_l2_reg
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name) )
            loss += l2
            return loss

    def _cross_entropy_iou_loss(self, bbox_pred, bbox_targets, name = "iou_loss"):
        with tf.name_scope(name):
            pred_w = tf.slice(bbox_pred,[0,0,2],[-1,-1,1])
            pred_h = tf.slice(bbox_pred,[0,0,3],[-1,-1,1])
            targ_w = tf.slice(bbox_targets,[0,0,2],[-1,-1,1])
            targ_h = tf.slice(bbox_targets,[0,0,3],[-1,-1,1])

            bbox_pred = \
                image_processing.bbox_tranformation_xywh_yxyx(bbox_pred)
            bbox_targets = \
                image_processing.bbox_tranformation_xywh_yxyx(bbox_targets)

            pred_ymin = tf.slice(bbox_pred,[0,0,0],[-1,-1,1])
            pred_xmin = tf.slice(bbox_pred,[0,0,1],[-1,-1,1])
            pred_ymax = tf.slice(bbox_pred,[0,0,2],[-1,-1,1])
            pred_xmax = tf.slice(bbox_pred,[0,0,3],[-1,-1,1])

            targ_ymin = tf.slice(bbox_targets,[0,0,0],[-1,-1,1])
            targ_xmin = tf.slice(bbox_targets,[0,0,1],[-1,-1,1])
            targ_ymax = tf.slice(bbox_targets,[0,0,2],[-1,-1,1])
            targ_xmax = tf.slice(bbox_targets,[0,0,3],[-1,-1,1])

            yA = tf.maximum(pred_ymin, targ_ymin)
            xA = tf.maximum(pred_xmin, targ_xmin)
            yB = tf.minimum(pred_ymax, targ_ymax)
            xB = tf.minimum(pred_xmax, targ_xmax)

            inter_Area = tf.maximum(0.0,(xB-xA))*tf.maximum(0.0,(yB-yA))
            pred_Area = pred_w*pred_h
            targ_Area = targ_w*targ_h
            iou = inter_Area / tf.clip_by_value((pred_Area + targ_Area - inter_Area),1e-10, 2)

            losses = -1.0 * tf.log(tf.clip_by_value(iou,1e-10,1))

            # inter_Area = tf.add(tf.multiply(tf.subtract(xB,xA), tf.subtract(yB,yA)), 1e-10)
            # pred_Area = tf.clip_by_value(tf.multiply(pred_w, pred_h), 1e-10, 1-1e-10)
            # targ_Area = tf.multiply(targ_w, targ_h)

            # losses = tf.multiply(tf.fill(tf.shape(inter_Area),-1.0),tf.log(tf.clip_by_value(inter_Area,1e-10,0.99)))

            # iou = tf.divide(inter_Area, tf.subtract(tf.add(pred_Area, targ_Area), inter_Area))
            # losses = tf.multiply(tf.fill(tf.shape(iou),-1.0),tf.log(tf.clip_by_value(iou,1e-10,1)))
            # losses = tf.clip_by_value(losses, 1e-10,2)

        return losses

    def _invalid_bbox_loss(self, bbox_pred, name="invalid_bbox_loss"):
        with tf.name_scope(name):
            bbox_pred_yxyx = \
                image_processing.bbox_tranformation_xywh_yxyx(bbox_pred)
            bbox_pred = tf.concat([bbox_pred, bbox_pred_yxyx], 2)
            center_l1_losses = tf.abs(bbox_pred - 0.5)
            self.center_l1_losses = center_l1_losses
            outer_pos = tf.cast(tf.less(0.5, center_l1_losses),tf.float32)
            outer_l1_losses = tf.multiply(center_l1_losses, outer_pos)
            total_loss = tf.reduce_sum(outer_l1_losses) / tf.cast(tf.size(bbox_pred),tf.float32)

            self.invalid_totla_loss = total_loss

        return total_loss

    def _invalid_bbox(self, bbox, name="invalid_bbox"):
        with tf.name_scope(name):
            greater_than_one = tf.cast(tf.greater(bbox,1),tf.float32)
            smaller_than_zero = tf.cast(tf.less(bbox,0),tf.float32)
            greater_than_one = (bbox-1)*greater_than_one
            smaller_than_zero = bbox*smaller_than_zero
            invalid_bbox_l2 = tf.square(greater_than_one + smaller_than_zero)
            return invalid_bbox_l2

            # bbox_to_center = tf.square(bbox-0.5)-0.25
            # mask = tf.cast(tf.greater(bbox_to_center,0),tf.float32)
            # invalid_bbox_l1 = tf.multiply(bbox_to_center,mask)
            # return invalid_bbox_l1

    def _invalid_area(self, bbox, name="invalid_bbox_area"):
        with tf.name_scope(name):
           gc = global_config.global_config
           w = tf.slice(bbox,[0,0,2],[-1,-1,1])
           h = tf.slice(bbox,[0,0,3],[-1,-1,1])
           bbox_area = tf.squeeze(w*h)
           greater_than_upper = tf.cast(tf.greater(bbox_area,gc.area_upper_bound),tf.float32)
           smaller_than_lower = tf.cast(tf.less(bbox_area,gc.area_lower_bound),tf.float32)
           greater_than_upper = (bbox_area-gc.area_upper_bound) * greater_than_upper
           smaller_than_lower = (gc.area_lower_bound-bbox_area) * smaller_than_lower
           invalid_area_l2 = tf.square(greater_than_upper + smaller_than_lower)
           return invalid_area_l2

    def bb_intersection_over_union(self, boxA, boxB):
        '''
        IOU exmaple
        '''
        #determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def get_iou(self, pred_bboxes, targ_bboxes, func_call='', mode=''):
        '''
        targ_bboxes (N,T,4) mscoco
        pred_bboxes (N,T,4) or
                    (N,4)-> expand_dim and broadcast, used to find max iou in target
                    faster-rcnn
        return iou (N,T,4)
        '''
        # pred_bboxes = tf.cond(tf.rank(pred_bboxes) < 3,
                    # lambda: tf.expand_dims(pred_bboxes,1),
                    # lambda: tf.identity(pred_bboxes))

        pred_w = tf.slice(pred_bboxes,[0,0,2],[-1,-1,1])
        pred_h = tf.slice(pred_bboxes,[0,0,3],[-1,-1,1])
        targ_w = tf.slice(targ_bboxes,[0,0,2],[-1,-1,1])
        targ_h = tf.slice(targ_bboxes,[0,0,3],[-1,-1,1])

        pred_bboxes =_debug_func(pred_bboxes ,'getiou_pred_bboxes_frcnn'+func_call,break_point=False, to_file=True)
        targ_bboxes =_debug_func(targ_bboxes ,'getiou_targ_bboxes_mscoco'+func_call,break_point=False, to_file=True)
        pred_bboxes = self._convert_coordinate(pred_bboxes, "frcnn","bmp",dim=3)
        targ_bboxes = self._convert_coordinate(targ_bboxes, "mscoco","bmp",dim=3)
        pred_bboxes =_debug_func(pred_bboxes ,'getiou_pred_bboxes_bmp'+func_call,break_point=False, to_file=True)
        targ_bboxes =_debug_func(targ_bboxes ,'getiou_targ_bboxes_bmp'+func_call,break_point=False, to_file=True)

        pred_xmin = tf.slice(pred_bboxes,[0,0,0],[-1,-1,1])
        pred_ymin = tf.slice(pred_bboxes,[0,0,1],[-1,-1,1])
        pred_xmax = tf.slice(pred_bboxes,[0,0,2],[-1,-1,1])
        pred_ymax = tf.slice(pred_bboxes,[0,0,3],[-1,-1,1])

        targ_xmin = tf.slice(targ_bboxes,[0,0,0],[-1,-1,1])
        targ_ymin = tf.slice(targ_bboxes,[0,0,1],[-1,-1,1])
        targ_xmax = tf.slice(targ_bboxes,[0,0,2],[-1,-1,1])
        targ_ymax = tf.slice(targ_bboxes,[0,0,3],[-1,-1,1])

        xmin_mask = tf.cast(tf.less_equal(pred_xmin, targ_xmin),tf.float32)
        ymin_mask = tf.cast(tf.less_equal(pred_ymin, targ_ymin),tf.float32)
        xmax_mask = tf.cast(tf.greater_equal(pred_xmax, targ_xmax),tf.float32)
        ymax_mask = tf.cast(tf.greater_equal(pred_ymax, targ_ymax),tf.float32)
        mask = xmin_mask*ymin_mask*xmax_mask*ymax_mask

        yA = tf.maximum(pred_ymin, targ_ymin)
        xA = tf.maximum(pred_xmin, targ_xmin)
        yB = tf.minimum(pred_ymax, targ_ymax)
        xB = tf.minimum(pred_xmax, targ_xmax)

        inter_Area = tf.maximum(0.0,(xB-xA))*tf.maximum(0.0,(yB-yA))
        pred_Area = pred_w*pred_h
        targ_Area = targ_w*targ_h
        inter_Area =_debug_func(inter_Area ,'getiou_inter_Area'+func_call,break_point=False, to_file=True)
        pred_Area =_debug_func(pred_Area ,'getiou_pred_Area'+func_call,break_point=False, to_file=True)
        targ_Area =_debug_func(targ_Area ,'getiou_targ_Area'+func_call,break_point=False, to_file=True)
        iou = inter_Area / tf.clip_by_value((pred_Area + targ_Area - inter_Area),1e-10, 2)

        if mode == 'embody':
            iou = iou*mask

        return iou

