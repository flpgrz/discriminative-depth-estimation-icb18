from __future__ import division, print_function
import cv2
from torch.utils.data import Dataset
import numpy as np
import random
import glob
from os.path import join, basename, dirname, exists
import pickle
from PIL import Image

from face_verification.utilities import normalization, image_processing

bad_test = np.array([
    (48, 140, 319),
    (48, 764, 809),
    (49, 0, 1420),
    (66, 623, 674),
    (66, 810, 822),
    (69, 0, 1403),
    (70, 1623, 1641),
    (70, 1698, 1709),
    (70, 2010, 2020),
    (80, 710, 730),
    (80, 932, 980),
    (96, 396, 426),
    (96, 1416, 1437),
    (96, 1817, 1859),
    (96, 1896, 1935),
    (97, 39, 57),
    (97, 113, 139),
    (97, 173, 206),
    (97, 809, 837),
    (97, 1102, 1127),
    (98, 112, 140),
    (98, 319, 350),
    (98, 386, 423),
    (98, 512, 551),
    (98, 687, 712),
    (98, 836, 861),
    (98, 1434, 1484),
    (98, 1527, 1605),
    (100, 18, 44),
    (100, 260, 315),
    (100, 386, 424),
    (100, 593, 602),
    (100, 1006, 1042),
    (100, 1096, 1121),
    (100, 1287, 1371),
    (100, 1822, 1845)
])


def random_mask(x, mask_percent=0.2):
    r, c = x.size

    r_i = int(random.randint(0, r - int(r * mask_percent) - 1))
    r_e = int(r_i + r * mask_percent)

    c_i = int(random.randint(0, c - int(c * mask_percent) - 1))
    c_e = int(c_i + c * mask_percent)

    x = np.array(x)
    x[r_i:r_e, c_i:c_e] = np.random.rand(int(r * mask_percent), int(c * mask_percent))

    return Image.fromarray(x)


class PandoraDataset(Dataset):
    def __init__(self, dataset_path, image_type=0, included_seq_type=(0, 1, 2), dict_filename=None,
                 filter_head_pose=False, filter_max_angle=10.0, filter_min_angle=-10.0, filter_type='inclusive',
                 sampling_rate=1, phase='train', validation_ids=(8, 17, 20, 21), test_ids=(9, 13, 15, 19),
                 data_augmentation=False, normalization_type=-1, preprocessing_type=-1):

        self.dataset_path = dataset_path
        self.image_type = image_type
        self.included_seq_type = included_seq_type
        self.data_augmentation = data_augmentation
        self.normalization_type = normalization_type
        self.preprocessing_type = preprocessing_type
        self.phase = phase
        self.validation_ids = validation_ids
        self.test_ids = test_ids

        if dict_filename is not None:
            dict_filename += '_' + str(sampling_rate) + '_' + self.phase + '.pkl'

        if dict_filename is not None and exists(dict_filename):
            print("Loading dictionary..."),
            with open(dict_filename, 'rb') as fd:
                self.d, self.l = pickle.load(fd)
        else:
            print("Creating Pandora dictionary..."),

            list_names = sorted(glob.glob(join(dataset_path, "*/*.txt")))

            self.d = dict()
            self.l = list()
            # self.dataset_len = 0

            for name in list_names:

                f = open(name, "r")
                lines = f.readlines()[::sampling_rate]

                seq = int(basename(dirname(name)))
                id = int((seq - 1)) // 5

                seq = (int(basename(dirname(name))) - 1) % 5

                if self.phase == 'train':
                    if id in self.validation_ids or id in self.test_ids:
                        continue
                elif self.phase == 'val':
                    if id not in self.validation_ids:
                        continue
                elif self.phase == 'test':
                    if id not in self.test_ids:
                        continue

                if seq not in self.included_seq_type:
                    continue

                if id not in self.d:
                    self.d[id] = dict()

                if seq not in self.d[id]:
                    self.d[id][seq] = list()

                # self.dataset_len += len(lines)

                for line in lines:
                    items = line[:-1].split("\t")

                    num_frame = int(items[0])
                    roll = float(items[1])
                    pitch = float(items[2])
                    yaw = float(items[3])

                    if self.image_type == 0:
                        filename = "frame_%06d_face_depth.png" % num_frame
                    elif self.image_type == 1:
                        filename = "frame_%06d_face_rgb.png" % num_frame
                    elif self.image_type == 2:
                        filename = "frame_%06d_face_rgb.png" % num_frame
                    elif self.image_type == 3:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

                    frame_dict = {
                        'id': id, 'seq': seq, 'frame': num_frame, 'filename': join(dirname(name), filename),
                        'roll': roll, 'pitch': pitch, 'yaw': yaw,
                        'img': cv2.imread(join(dirname(name), filename), cv2.IMREAD_ANYDEPTH).astype(np.uint8)
                    }

                    # filter head angles
                    if filter_head_pose:
                        max_angle = filter_max_angle
                        min_angle = filter_min_angle
                        if filter_type == 'inclusive':
                            if (min_angle < roll < max_angle) and (min_angle < pitch < max_angle) and (
                                    min_angle < yaw < max_angle):
                                self.d[id][seq].append(frame_dict)
                                self.l.append(frame_dict)
                        elif filter_type == 'exclusive':
                            if (roll < min_angle or roll > max_angle) or (pitch < min_angle or pitch > max_angle) or (
                                    yaw < min_angle or yaw > max_angle):
                                self.d[id][seq].append(frame_dict)
                                self.l.append(frame_dict)
                        elif filter_type == 'extreme':
                            if (roll < min_angle or roll > max_angle) and (pitch < min_angle or pitch > max_angle) and (
                                    yaw < min_angle or yaw > max_angle):
                                self.d[id][seq].append(frame_dict)
                                self.l.append(frame_dict)
                        else:
                            raise NotImplementedError
                    else:
                        self.d[id][seq].append(frame_dict)
                        self.l.append(frame_dict)

            if dict_filename is not None:
                with open(dict_filename, 'wb') as fd:
                    pickle.dump((self.d, self.l), fd, protocol=pickle.HIGHEST_PROTOCOL)

        self.train_da = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-5, +5)),
            # transforms.Lambda(random_mask)
        ])

        self.val_da = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((100, 100)),
        ])

        self.test_da = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((100, 100)),
        ])

        print("Dictionary created/loaded.")

    def __getitem__(self, index):

        current_frame = self.l[index]
        current_seq = current_frame['seq']
        current_person_id = current_frame['id']

        same = random.randint(0, 1)
        # ToDo generare due coppie (una positiva e una negativa) invece che una sola

        if same:
            img1 = current_frame['img'].copy()
            it = 0
            while True:
                seq_id = list(self.d[current_person_id].keys())[
                    random.randint(0, len(self.d[current_person_id].keys()) - 1)]
                x = self.d[current_person_id][seq_id]
                it += 1
                if len(x) > 0 or it == 25:
                    break
            if len(x) == 0 and it == 25:
                raise Exception('Unable to find a valid sequence for the person %d' % current_person_id)
            other_frame = x[random.randint(0, len(x) - 1)]
            img2 = other_frame['img'].copy()
            names = self.get_names(same, current_frame, other_frame)

        else:
            img1 = current_frame['img'].copy()
            different_person_id = current_person_id
            while different_person_id == current_person_id:
                different_person_id = list(self.d.keys())[random.randint(0, len(self.d.keys()) - 1)]
            it = 0
            while True:
                seq_id = list(self.d[different_person_id].keys())[
                    random.randint(0, len(self.d[different_person_id].keys()) - 1)]
                x = self.d[different_person_id][seq_id]
                it += 1
                if len(x) > 0 or it == 25:
                    break
            if len(x) == 0 and it == 25:
                raise Exception('Unable to find a valid sequence for the person %d' % different_person_id)
            other_frame = x[random.randint(0, len(x) - 1)]
            img2 = other_frame['img'].copy()
            names = self.get_names(same, current_frame, other_frame)

        if self.data_augmentation:
            img1 = np.expand_dims(img1, 2)
            img2 = np.expand_dims(img2, 2)

            if self.phase == 'train':
                img1 = self.train_da(img1)
                img2 = self.train_da(img2)
            elif self.phase == 'val':
                img1 = self.val_da(img1)
                img2 = self.val_da(img2)
            elif self.phase == 'test':
                img1 = self.test_da(img1)
                img2 = self.test_da(img2)

        img1 = np.asarray(img1, dtype=np.float32)
        img2 = np.asarray(img2, dtype=np.float32)

        # pre-processing
        img1, img2 = image_processing(img1, img2, self.preprocessing_type)

        # normalization
        img1, img2 = normalization(img1, img2, self.normalization_type)

        img1 = np.expand_dims(img1, 0)
        img2 = np.expand_dims(img2, 0)

        return img1, img2, same, names, current_frame['filename'], other_frame['filename']

    def __len__(self):
        return len(self.l)

    def get_names(self, same, current_frame, other_frame):
        if same:
            same = 'Same'
        else:
            same = 'Different'

        img1_name = '\n🚀 Img1 {' + str(same) + '_ID_' + str(current_frame['id']) + '_Seq_' + str(
            current_frame['seq']) + '_Frame_' + str(current_frame['frame']) + '} '
        img2_name = 'Img2 {' + str(same) + '_ID_' + str(other_frame['id']) + '_Seq_' + str(
            other_frame['seq']) + '_Frame_' + str(other_frame['frame']) + '} 🚀\n'
        names = img1_name + img2_name
        return names


class PandoraDatasetTestFixed(Dataset):
    def __init__(self, dataset_path, target_size, image_type=1, len_subset=-1, correct_dataset_path=None,
                 return_filenames=False, dataset_name='pandora'):

        self.dataset_path = dataset_path
        self.target_size = target_size
        self.image_type = image_type
        self.len_subset = len_subset
        self.correct_dataset_path = correct_dataset_path
        self.return_filenames = return_filenames
        self.dataset_name = dataset_name

        if self.dataset_name == "pandora":
            print("Loading Pandora fixed couples...")
        elif self.dataset_name == 'BIWI':
            print("Loading Biwi fixed couples...")
        else:
            raise NotImplementedError

        with open(dataset_path, "r") as fd:
            self.file_content = fd.readlines()

        if len_subset != -1:
            random.shuffle(self.file_content)
            self.file_content = self.file_content[:len_subset]

        if self.dataset_name == 'pandora':
            # ToDo Correggere - Al momento utilizziamo un subset "semplificato" di immagini durante il training ed il testing della rete
            self.file_content_ok = list()
            for i, fc in enumerate(self.file_content):
                elements = fc.split("\t")
                filename_img1 = elements[0]
                filename_img2 = elements[1]
                same = int(elements[2].strip())

                frame1 = int(basename(filename_img1)[6:12])
                frame2 = int(basename(filename_img2)[6:12])
                seq1 = int(basename(dirname(filename_img1)))
                seq2 = int(basename(dirname(filename_img2)))

                exc1 = bad_test[bad_test[:, 0] == seq1]
                exc2 = bad_test[bad_test[:, 0] == seq2]

                excluded = False
                for e in exc1:
                    if e[1] <= frame1 <= e[2]:
                        excluded = True
                        break
                for e in exc2:
                    if e[1] <= frame2 <= e[2]:
                        excluded = True
                        break

                if not excluded:
                    self.file_content_ok.append(fc)

            self.file_content = self.file_content_ok

        print("Couples loaded.")

    def __getitem__(self, index):

        current_line = self.file_content[index]
        elements = current_line.split("\t")

        filename_img1 = elements[0]
        filename_img2 = elements[1]
        same = int(elements[2].strip())

        if self.correct_dataset_path is not None:
            filename_img1 = filename_img1.replace('/datasets', self.correct_dataset_path)
            filename_img2 = filename_img2.replace('/datasets', self.correct_dataset_path)

        if self.image_type == 0:
            pass
        elif self.image_type == 1:
            filename_img1 = filename_img1.replace('BIWI_face_dataset_large', 'Biwi/face_dataset_largeRGB')
            filename_img2 = filename_img2.replace('BIWI_face_dataset_large', 'Biwi/face_dataset_largeRGB')
            filename_img1 = filename_img1.replace('face_dataset_16', 'face_dataset_gray')
            filename_img2 = filename_img2.replace('face_dataset_16', 'face_dataset_gray')
            filename_img1 = filename_img1.replace('_depth', '_rgb')
            filename_img2 = filename_img2.replace('_depth', '_rgb')
        elif self.image_type == 2:
            filename_img1 = filename_img1.replace('face_dataset_16', 'face_dataset_RGB')
            filename_img2 = filename_img2.replace('face_dataset_16', 'face_dataset_RGB')
            filename_img1 = filename_img1.replace('_depth', '_rgb')
            filename_img2 = filename_img2.replace('_depth', '_rgb')
        elif self.image_type == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.image_type == 0:
            img1 = cv2.imread(filename_img1, cv2.IMREAD_ANYDEPTH).astype(np.uint8)
            img2 = cv2.imread(filename_img2, cv2.IMREAD_ANYDEPTH).astype(np.uint8)
        elif self.image_type == 1:
            img1 = cv2.imread(filename_img1, 0)
            img2 = cv2.imread(filename_img2, 0)
        elif self.image_type == 2:
            img1 = cv2.imread(filename_img1, 1)
            img2 = cv2.imread(filename_img2, 1)
        else:
            raise NotImplementedError

        img1 = cv2.resize(img1, (self.target_size, self.target_size))
        img2 = cv2.resize(img2, (self.target_size, self.target_size))

        img1 = np.expand_dims(img1, 0)
        img2 = np.expand_dims(img2, 0)

        if self.image_type == 0:
            img1 = (img1.astype(np.float32) - 117.5) / 117.5
            img2 = (img2.astype(np.float32) - 117.5) / 117.5
        elif self.image_type == 1:
            img1 = (img1.astype(np.float32) - 127.5) / 127.5
            img2 = (img2.astype(np.float32) - 127.5) / 127.5
        elif self.image_type == 2:
            img1 = (img1.astype(np.float32) - 127.5) / 127.5
            img2 = (img2.astype(np.float32) - 127.5) / 127.5
        else:
            raise NotImplementedError

        if self.return_filenames:
            return img1, img2, same, filename_img1, filename_img2
        else:
            return img1, img2, same

    def __len__(self):
        return len(self.file_content)
