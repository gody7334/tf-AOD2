import numpy as np
import cPickle as pickle
import hickle
import time
import os
from utils.config import global_config

def load_coco_data(data_path='./data/data', split='train', if_train=True, part=0):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}

    data['features'] = hickle.load(os.path.join(
        data_path, '%s.features.part%d.hkl' % (split,part)))
    data['images'] = hickle.load(os.path.join(
        data_path, '%s.images.part%d.hkl' % (split,part)))

    with open(os.path.join(data_path, '%s.file.names.part%d.pkl' % (split,part)), 'rb') as f:
        data['file_names'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.bboxes.part%d.pkl' % (split,part)), 'rb') as f:
        data['bboxes'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.classes.part%d.pkl' % (split,part)), 'rb') as f:
        data['classes'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.part%d.pkl' % (split,part)), 'rb') as f:
        data['image_idxs'] = pickle.load(f)

    gc = global_config.global_config
    bboxes_area = data['bboxes'][:,:,2]*data['bboxes'][:,:,3]
    bboxes_area_zero_mask = bboxes_area == 0
    bboxes_area_threshlod_mask = (bboxes_area > gc.area_lower_bound) * (bboxes_area <= gc.area_upper_bound)
    bboxes_mask = np.prod(bboxes_area_threshlod_mask + bboxes_area_zero_mask, axis=1)
    bboxes_index = np.argwhere(bboxes_mask==1)

    data['file_names'] = np.squeeze(data['file_names'][bboxes_index])
    data['bboxes'] = np.squeeze(data['bboxes'][bboxes_index])
    data['classes'] = np.squeeze(data['classes'][bboxes_index])
    data['image_idxs'] = np.squeeze(data['image_idxs'][bboxes_index])

    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" % (end_t - start_t)
    return data

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names


def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' % (epoch + 1))
        f.write('Bleu_1: %f\n' % scores['Bleu_1'])
        f.write('Bleu_2: %f\n' % scores['Bleu_2'])
        f.write('Bleu_3: %f\n' % scores['Bleu_3'])
        f.write('Bleu_4: %f\n' % scores['Bleu_4'])
        f.write('METEOR: %f\n' % scores['METEOR'])
        f.write('ROUGE_L: %f\n' % scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' % scores['CIDEr'])


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' % path)
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)
