from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf

# get parsing argument
from utils.config import global_config
from utils.config.config import cfg
import utils.roi_data_layer.roidb as rdl_roidb

from utils.dataset.factory import get_imdb
import utils.dataset.imdb

class PrepareDataset(object):

    def __init__(self):
        # imdb only contain dataset information, like class index, data index
        # roidb contain label, like class label, bounding box,....
        self.imdb = None
        self.roidb = None

        self.combined_roidb('voc_2007_trainval')

    def combined_roidb(self, imdb_names):
        """
        Combine multiple roidbs
        """
        def get_roidb(imdb_name):
            imdb = get_imdb(imdb_name)
            # import ipdb; ipdb.set_trace()
            print('Loaded dataset `{:s}` for training'.format(imdb.name))
            imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
            print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
            roidb = self.get_training_roidb(imdb)
            return roidb

        roidbs = [get_roidb(s) for s in imdb_names.split('+')]
        roidb = roidbs[0]
        # from pprint import pprint as pp
        # import ipdb; ipdb.set_trace()
        if len(roidbs) > 1:
            for r in roidbs[1:]:
              roidb.extend(r)
            tmp = get_imdb(imdb_names.split('+')[1])
            imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
        else:
            imdb = get_imdb(imdb_names)

        self.imdb = imdb
        self.roidb = roidb

    def get_training_roidb(self, imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing training data...')
        rdl_roidb.prepare_roidb(imdb)
        print('done')

        return imdb.roidb
