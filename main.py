import _init_paths
from control.solver import Solver
from core.load_dataset import load_coco_data

import os, shutil
import ipdb
from random import randint
import os.path

from utils.config import global_config
from graph.forward.aod import AOD
from graph.backward import Backward

import colored_traceback.always

def main():
    prepare()

    if global_config.global_config.mode == 'train':
        model =AOD(mode='train',data=None)
        optimizer = Backward(model = model)
        solver = Solver(model, optimizer, None, None)
        while(True):
            part = randint(0,128)
            if os.path.isfile(os.path.join('./data/data/train', '%s.file.names.part%d.pkl' % ('train',part))):
                solver.data = load_coco_data(data_path='./data/data', split='train', if_train=True, part=part)
                solver.val_data = load_coco_data(data_path='./data/data', split='val', if_train=False, part=part)
                if ((len(solver.data) > 0) and (len(solver.val_data))):
                    solver.train()
                    solver.val()

    elif global_config.global_config.mode == 'val':
        data = None
        val_data = load_coco_data(data_path='./data/data', split='val', if_train=False)
        model =AOD(mode='val',data=None)
        solver = Solver(model, None, data, val_data)
        solver.val()

def prepare():
    global_config.assign_config()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # or any {'0' default, '1' info, '2' warning, '3' error}

    if global_config.global_config.device == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif global_config.global_config.device == "gpu0":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif global_config.global_config.device == "gpu1":
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    clean_folder(global_config.global_config.tf_log_dir)

def clean_folder(folder_dir):
    for the_file in os.listdir(folder_dir):
        file_path = os.path.join(folder_dir, the_file)
        rm_file(file_path)

def rm_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
