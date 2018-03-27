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
from graph.forward.Iforward import IForward
from graph.ops.debug import _debug_func

class LSTM_Attension(IForward):
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
        self.T = 4 # Time step size of LSTM (how many predicting bboxes in a image)


        # Place holder for features and captions
        # self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        # self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.features = None
        self.captions = None
        self.logits = None
        self.targets = None
        self.weights = None

        self.build_image_embeddings()
        self.build_model()

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.
           pass image into inceptionV3 and get image features map (add full connected layer at the end)
        Inputs:
          self.images

        Outputs:
          self.image_embeddings
        """
        # call parent function to get original image features map
        super(LSTM_Attension, self).build_image_embeddings()

        # TOOD experiement different layer
        inception_layer = self.inception_end_points['Mixed_7c']
        # get depth of image embedded
        layer_shape = inception_layer.get_shape().as_list()
        self.D = layer_shape[3]
        self.L = layer_shape[1]*layer_shape[2]
        # flatten image pixel from 2D to 1D
        self.image_embeddings = tf.reshape(inception_layer, [self.config.batch_size, self.L, self.D])
        self.features = self.image_embeddings

    def build_seq_embeddings(self):
        return

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)

            return c, h

    # def _word_embedding(self, inputs, reuse=False):
    #     with tf.variable_scope('word_embedding', reuse=reuse):
    #         w = tf.get_variable('w', [self.V, self.M],initializer=self.emb_initializer)
    #         x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
    #         return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D],initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, logits, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D],initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            w_logit = tf.get_variable('w_logit', [4, self.L],initializer=self.weight_initializer)
            b_logit = tf.get_variable('b_logit', [self.L], initializer=self.const_initializer)

            pix_bbox = tf.nn.relu(tf.matmul(logits, w_logit) + b_logit)    # (N, L)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L]) + pix_bbox  # (N, L)
            # out_att = pix_bbox
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, 4], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [4], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [4, 4], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [4], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, 4], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self):
        features = self.features

        batch_size = tf.shape(self.features)[0]

        captions_in = self.input_seqs
        captions_out =  self.target_seqs
        mask = self.input_mask

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')
        c, h = self._get_initial_lstm(features=features)
        # x = self._word_embedding(inputs=captions_in)
        x = self.input_seqs
        x.set_shape([self.config.batch_size, None, 4])

        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        logit_list = []
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        logits = tf.fill([self.config.batch_size,4],0.0)

        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, logits, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context], 1), state=[c, h])

            logits = self._decode_lstm(x[:, t, :], h, context, dropout=self.dropout, reuse=(t != 0))
            logit_list.append(logits)
            # loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, captions_out[:, t]) * mask[:, t])

        logits = tf.transpose(tf.stack(logit_list),(1,0,2))
        self.logits = logits

        image_processing.draw_tf_bounding_boxes(self.images, logits, name="logits")
        image_processing.draw_tf_bounding_boxes(self.images, self.target_seqs, name="targets")

        # get lstm output shape (batch_size, length of bboxes per image, lstm output dim)
        # lstm_outputs_shape = tf.shape(logits)

        # logits_origin_shape = tf.concat(axis=0,
                                        # values=[tf.cast(tf.slice(lstm_outputs_shape,[0],[2]),tf.int32),
                                        # tf.constant(4,tf.int32,[1])])

        # weights = tf.cast(tf.reshape(mask, logits_origin_shape),tf.float32)
        weights = tf.cast(tf.slice(mask,[0,0,0],[-1,self.T,-1]),tf.float32)
        self.weights = weights
        reduce_weights = tf.reduce_mean(weights, axis=2)
        reduce_weights = tf.slice(reduce_weights,[0,0],[-1,self.T])

        # Compute losses.
        # loss for target x y w h
        l1_losses = self._smooth_l1_loss(bbox_pred=logits, bbox_targets=tf.slice(captions_out,[0,0,0],[-1,self.T,-1]))
        l1_losses = tf.reshape(l1_losses,[-1])
        weights = tf.reshape(weights, [-1])
        batch_l1_loss = tf.div(tf.reduce_sum(tf.multiply(l1_losses, weights)),
                             tf.reduce_sum(weights), name="batch_l1_loss")

        # loss for invalid logit xmin ymin xmax ymax w h
        invalid_loss = self._invalid_bbox_loss(logits)

        # loss for IoU to target bbox
        iou_losses = self._cross_entropy_iou_loss(logits, tf.slice(captions_out,[0,0,0],[-1,self.T,-1]))
        iou_losses = tf.reshape(iou_losses,[-1])
        reduce_weights = tf.reshape(reduce_weights, [-1])

        iou_loss = tf.div(tf.reduce_sum(tf.multiply(iou_losses, reduce_weights)),
                          tf.reduce_sum(reduce_weights), name="iou_loss")


        batch_loss = tf.cond(invalid_loss > 0,
                lambda: invalid_loss*10+tf.stop_gradient(iou_loss+batch_l1_loss*100),
                lambda: invalid_loss*10+iou_loss+batch_l1_loss*100)
        # # batch_loss = invalid_loss + iou_loss
        # batch_loss = iou_loss
        # batch_loss = batch_l1_loss

        # batch_loss = self._l2_regularization(batch_loss)

        # losses = self._smooth_l1_loss(bbox_pred=logits, bbox_targets=tf.slice(captions_out,[0,0,0],[-1,self.T,-1]))
        # losses = self._get_min_smooth_l1_loss(bbox_pred=logits, bbox_targets=tf.slice(captions_out,[0,0,0],[-1,self.T,-1]))


        # batch_loss = tf.div(tf.reduce_sum(tf.multiply(tf.reshape(losses,[-1]), weights)),
                          # tf.reduce_sum(weights), name="batch_loss")
        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        # if self.alpha_c > 0:
        #     alphas = tf.transpose(tf.stack(alpha_list),(1, 0, 2))     # (N, T, L)
        #     alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
        #     alpha_reg = self.alpha_c * tf.reduce_sum((16. / 196 - alphas_all) ** 2)
        #     loss += alpha_reg

        # loss = loss / tf.to_float(batch_size)

         # Add summaries.
        tf.summary.scalar("losses/batch_loss", batch_loss)
        tf.summary.scalar("losses/total_loss", total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

        self.total_loss = total_loss
        self.target_smooth_l1_losses = iou_losses  # Used in evaluation.
        self.target_smooth_l1_losses_weights = weights  # Used in evaluation.

        return loss
