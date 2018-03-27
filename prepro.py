from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.inceptionV3 import *
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os, sys
import json

from pprint import pprint as pp
import colored_traceback.always
import ipdb

def _process_instance_data(instance_file, image_dir, max_length):
    with open(instance_file) as f:
        instance_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']:
            {'file_name':image['file_name'], 'height':image['height'], 'width':image['width']}
            for image in instance_data['images']}

    # data is a list of dictionary which contains 'segment', 'bbox', 'area', 'file_name' and 'image_id' as key.
    data = []
    for annotation in instance_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id]['file_name'])
        annotation['height'] = id_to_filename[image_id]['height']
        annotation['width'] = id_to_filename[image_id]['width']
        data += [annotation]

    instances = {}
    for d in data:
        inst = instances.get(d['image_id'])
        if inst is None:
            inst = {'width':'', 'height':'', 'area': [], 'bbox': [], 'category_id': [], 'file_name': '', 'image_id': '', 'iscrowd': [], 'segmentation': []}
        inst['width'] = d['width']
        inst['height'] = d['height']
        inst['area'].append(d['area'])
        inst['bbox'].append(d['bbox'])
        inst['category_id'].append(d['category_id'])
        inst['file_name'] = d['file_name']
        inst['image_id'] = d['image_id']
        inst['iscrowd'].append(d['iscrowd'])
        # inst['segmentation'].append(d['segmentation'])
        instances[d['image_id']] = inst

    data = []
    for key in instances.iterkeys():
        data.append(instances[key])

    # convert to pandas dataframe (for later visualization or debugging)
    instance_data = pd.DataFrame.from_dict(data)
    instance_data.sort_values(by='image_id', inplace=True)
    instance_data = instance_data.reset_index(drop=True)

    # del_idx = []
    # for i, bbox in enumerate(instance_data['bbox']):
        # caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        # caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        # caption = " ".join(caption.split())  # replace multiple spaces

        # caption_data.set_value(i, 'caption', caption.lower())
        # if len(caption.split(" ")) > max_length:
            # del_idx.append(i)

    # delete captions if size is larger than max_length
    # print "The number of captions before deletion: %d" %len(caption_data)
    # caption_data = caption_data.drop(caption_data.index[del_idx])
    # caption_data = caption_data.reset_index(drop=True)
    # print "The number of captions after deletion: %d" %len(caption_data)
    return instance_data


def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" %len(caption_data)
    return caption_data

def _get_bboxes_max_length(annotations):
    max_len = 0
    for i, bbox in enumerate(annotations['bbox']):
        if len(bbox) > max_len:
            max_len = len(bbox)
    return max_len

def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

def _build_bbox_vector(annotations, max_length=30):
    n_examples = len(annotations)
    bboxes = np.ndarray((n_examples,max_length,4)).astype(np.float32)
    for i in range(len(annotations)):
        # pad short bbox with the special [0,0,0,0] to make it fixed-size vector
        bbox = annotations.loc[i,'bbox']
        w = annotations.loc[i,'width']
        h = annotations.loc[i,'height']

        bbox_length = len(bbox)
        if bbox_length < (max_length):
            for j in range(max_length - bbox_length):
                bbox.append([0.0,0.0,0.0,0.0])
        bbox = np.asarray(bbox)
        scale = np.array([1/float(w),1/float(h),1/float(w),1/float(h)])
        bbox = bbox * scale
        bboxes[i, :] = np.asarray(bbox)
    print "Finished building bbox vectors"

    classes = np.ndarray((n_examples,max_length)).astype(np.float32)
    for i, c in enumerate(annotations['category_id']):
        # pad short class with 100 to make it fixed-size vector
        class_length = len(c)
        if class_length < (max_length):
            for j in range(max_length - class_length):
                c.append(100)
        classes[i, :] = np.asarray(c)
    print "Finished building class vectors"

    return (bboxes,classes)

def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 10
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path
    vgg_model_path = '/dataset/mscoco/tf-im2txtatten/data/imagenet-vgg-verydeep-19.mat'

    caption_file = 'data/data/annotations/captions_train2014.json'
    image_dir = 'image/%2014_resized/'

    reload_json = False
    if(reload_json):
        # about 80000 images
        instance_file = 'instances_train2014.json'
        # instance_file='/dataset/mscoco/raw-data/annotations/instances_train2014.json'
        train_dataset = _process_instance_data(instance_file=instance_file,
                                              image_dir='data/image/train2014_resized/',
                                              max_length=max_length)
        # about 40000 images
        instance_file = 'instances_val2014.json'
        # instance_file='/dataset/mscoco/raw-data/annotations/instances_val2014.json'
        val_dataset = _process_instance_data(instance_file=instance_file,
                                              image_dir='data/image/val2014_resized/',
                                              max_length=max_length)

        # about 800, 400 and 400 images for train / val / test dataset
        train_cutoff = int(0.01*len(train_dataset))
        val_cutoff = int(0.01 * len(val_dataset))
        test_cutoff = int(0.02 * len(val_dataset))
        print 'Finished processing instance data'

        save_pickle(train_dataset[:train_cutoff], 'data/data/train/train.annotations.pkl')
        save_pickle(val_dataset[:val_cutoff], 'data/data/val/val.annotations.pkl')
        save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/data/test/test.annotations.pkl')

    for split in ['train', 'val', 'test']:
        annotations = load_pickle('./data/data/%s/%s.annotations.pkl' % (split, split))

        max_length = _get_bboxes_max_length(annotations=annotations)
        (bboxes,classes) = _build_bbox_vector(annotations=annotations, max_length=max_length)
        save_pickle(bboxes, './data/data/%s/%s.bboxes.pkl' % (split, split))
        save_pickle(classes, './data/data/%s/%s.classes.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './data/data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, './data/data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_bboxes = {}
        i = -1
        for bbox, image_id in zip(annotations['bbox'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_bboxes[i] = []
            feature_to_bboxes[i].append(bbox)
        save_pickle(feature_to_bboxes, './data/data/%s/%s.references.pkl' % (split, split))
        print "Finished building %s bbox dataset" %split

    is_exit = False
    if is_exit is True:
        sys.exit()

    # build computation graph
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], 'images')
    inception_output, inception_end_points = inception_v3(
        images, trainable=False, is_training=False)
    inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
    features = inception_end_points['Mixed_5c']
    feature_shape = features.get_shape().as_list()
    NN = feature_shape[0]
    WW = feature_shape[1]
    HH = feature_shape[2]
    DD = feature_shape[3]

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # vggnet = Vgg19(vgg_model_path)
    # vggnet.build()
    with tf.Session() as sess:
        # initial variable and restore variable from checkpoint
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(inception_variables)
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                "/dataset/pretrain-model/im2txt/inception_v3.ckpt")
        saver.restore(sess, "/dataset/pretrain-model/im2txt/inception_v3.ckpt")

        for split in ['train', 'val', 'test']:
            anno_path = './data/data/%s/%s.annotations.pkl' % (split, split)
            save_path = './data/data/%s/%s.features.hkl' % (split, split)
            i_path = './data/data/%s/%s.images.hkl' % (split, split)

            annotations = load_pickle(anno_path)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)
            # n_examples = (n_examples/200)*10

            origin_images = np.ndarray([n_examples, 299, 299, 3], dtype=np.float32)
            all_feats = np.ndarray([n_examples, WW, HH, DD], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(features, feed_dict={images: image_batch})
                origin_images[start:end, :] = image_batch
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(origin_images, i_path)
            print ("Saved %s.." % (i_path))
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))



if __name__ == "__main__":
    main()
