from __future__ import division, print_function
import progress.bar
import numpy as np
from sklearn import preprocessing
from skimage import exposure
from PIL import Image
import random
import cv2
import os


def chunk(list_object, chunk_size):
    """Yield successive n-sized chunks from list_object."""
    for i in range(0, len(list_object), chunk_size):
        yield list_object[i:i + chunk_size]


class Bar(progress.bar.Bar):
    message = ''
    suffix = '%(percent)3d%% (%(index)d/%(max)d) - eps: %(elapsed_td)s - eta: %(eta_td)s - avg: %(avg).1fs'
    bar_prefix = ' |'
    bar_suffix = '| '
    empty_fill = '□'
    fill = '■'


class IncrementalBar(Bar, progress.bar.IncrementalBar):
    # message = ''
    # suffix = '%(percent)3d%% (%(index)d/%(max)d) - elapsed: %(elapsed_td)s - eta: %(eta_td)s - Avg: %(avg).1fs'
    # bar_prefix = ' |'
    # bar_suffix = '| '
    empty_fill = '▫'
    fill = '■'
    phases = ('▫', '▪', '■')


def get_hrr_id(name, join_glasses=False):
    name = os.path.basename(name)
    if join_glasses:
        id = name[0:2]
    else:
        name = name[:name.rfind('_')]
        if len(name) == 2:
            id = name
        else:
            if name == '04_g':
                id = '19'
            elif name == '05_g':
                id = '20'
            elif name == '09_g':
                id = '21'
            elif name == '17_g':
                id = '22'
            else:
                raise Exception('Unrecognized identity with glasses `' + name + '`')
    return id


def image_processing(img1, img2, type):

    if type == 0:
        # images must be uint8 for equalization
        img1 = cv2.equalizeHist(img1.astype('uint8'))
        img2 = cv2.equalizeHist(img2.astype('uint8'))
        # return as float32
        img1 = np.asarray(img1, dtype=np.float32)
        img2 = np.asarray(img2, dtype=np.float32)

    if type == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img1 = clahe.apply(img1.astype('uint8'))
        img2 = clahe.apply(img2.astype('uint8'))
        # return as float32
        img1 = np.asarray(img1, dtype=np.float32)
        img2 = np.asarray(img2, dtype=np.float32)

    return img1, img2


def normalization(img1, img2, type):

    if type == 0:
        img1 = (img1 - np.mean(img1)) / np.std(img1)
        img2 = (img2 - np.mean(img2)) / np.std(img2)

    if type == 1:
        img1 = img1 - 127.5
        img2 = img2 - 127.5

    if type == 2:
        p2, p98 = 0, 255
        img1 = exposure.rescale_intensity(np.asarray(img1, dtype='float32'), in_range=(p2, p98), out_range=(0, 1))
        img1 = preprocessing.scale(img1.astype(np.float32))
        img2 = exposure.rescale_intensity(img2.astype('float32'), in_range=(p2, p98), out_range=(0, 1))
        img2 = preprocessing.scale(img2.astype(np.float32))

    if type == 3:

        w = img1.shape[1]
        h = img1.shape[0]
        l = 10

        offset_min = 20
        offset_max = 30

        center_value_img1 = np.mean(img1[h//2 - l//2:h//2 + l//2, w//2 - l//2:w//2 + l//2])
        center_value_img2 = np.mean(img2[h//2 - l//2:h//2 + l//2, w//2 - l//2:w//2 + l//2])

        min_value_img1 = center_value_img1 - offset_min
        min_value_img2 = center_value_img2 - offset_min

        max_value_img1 = center_value_img1 + offset_max
        max_value_img2 = center_value_img2 + offset_max

        img1 = exposure.rescale_intensity(img2.astype('float32'),
                                          in_range=(min_value_img1, max_value_img1), out_range=(0, 1))
        img1 = preprocessing.scale(img1.astype(np.float32))

        img2 = exposure.rescale_intensity(img2.astype('float32'),
                                          in_range=(min_value_img2, max_value_img2), out_range=(0, 1))
        img2 = preprocessing.scale(img2.astype(np.float32))

    return img1, img2


def hybrid_data_augmentation(img_a, img_b, is_train, degrees=(-5, +5)):
    img_a = Image.fromarray(img_a)
    img_b = Image.fromarray(img_b)

    if is_train:
        prob = random.random()
        if prob > 0.5:
            img_a.transpose(Image.FLIP_LEFT_RIGHT)
            img_b.transpose(Image.FLIP_LEFT_RIGHT)

        prob = random.random()
        if prob > 0.5:
            d = (random.random() - 0.5) * (degrees[1] - degrees[0])
            img_a.rotate(d)
            img_b.rotate(d)

        prob = random.random()
        if prob > 0.5:
            noise = np.random.randn(*img_a.size) * 3
            img_a += noise
            img_b += noise

    return img_a, img_b
