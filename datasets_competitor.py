import cv2
import numpy as np

from typing import *
from torch.utils.data import Dataset
from numpy import ndarray as NPArray
from pathlib import Path
import sklearn


class Faces(Dataset):
    def __init__(self, data_path: str, data_name: str, dataset_test_folder: bool = True,
                 target_size: Tuple[int, int] = (64, 64), training: bool = True,
                 return_filenames: bool = False):

        self.data_path = data_path
        self.target_size = target_size
        self.training = training
        self.data_name = data_name
        self.dataset_test_folder = dataset_test_folder
        self.return_filenames = return_filenames
        self.image_shape = (3,) + self.target_size

        if self.data_name == 'biwi':
            self.gt_test_names = np.load('biwi_file_names/biwi_input_test_names.npy')
            self.input_test_names = np.load('biwi_file_names/biwi_gt_test_names.npy')
            self.input_names = np.load('biwi_file_names/biwi_gt_names.npy')
            self.gt_names = np.load('biwi_file_names/biwi_input_names.npy')

        elif self.data_name == 'pandora':
            self.gt_names = np.load('pandora_file_names/pandora_gt_names.npy')
            self.input_names = np.load('pandora_file_names/pandora_input_names.npy')
            self.gt_test_names = np.load('pandora_file_names/pandora_gt_test_names.npy')
            self.input_test_names = np.load('pandora_file_names/pandora_input_test_names.npy')

        if training:
            self.dataset_size = len(self.input_names)
        else:
            self.dataset_size = len(self.input_test_names)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, i: int) -> Tuple[NPArray, NPArray]:
        if self.training:
            input_filename = str(self.input_names[i])
            gt_filename = str(self.gt_names[i])
        else:
            input_filename = str(self.input_test_names[i])
            gt_filename = str(self.gt_test_names[i])

        img_input = cv2.imread(input_filename, 0)
        img_input = cv2.resize(img_input, self.target_size[::-1], interpolation=cv2.INTER_CUBIC)
        img_input = (img_input.astype(np.float32) - 127.5) / 127.5
        img_input = np.expand_dims(img_input, 0)

        img_gt = cv2.imread(gt_filename, 0)
        img_gt = cv2.resize(img_gt, self.target_size[::-1], interpolation=cv2.INTER_CUBIC)
        img_gt = (img_gt.astype(np.float32) - 127.5) / 127.5
        img_gt = np.expand_dims(img_gt, 0)

        if self.data_name == 'pandora':
            gt_id_label = get_pandora_label(input_filename)
        elif self.data_name == 'biwi':
            gt_id_label = get_biwi_label(input_filename)


        return img_input, img_gt, gt_id_label

def get_pandora_label(filename):
    identity = int(filename.split('/')[5])
    # identity_one_hot = np.zeros(shape=(100,))
    # identity_one_hot[identity-1] = 1
    # identity_one_hot.astype(np.float32)
    return identity-1

def get_biwi_label(filename):
    identity = int(filename.split('/')[6])    # identity_one_hot = np.zeros(shape=(100,))
    # identity_one_hot[identity-1] = 1
    # identity_one_hot.astype(np.float32)
    return identity-1
