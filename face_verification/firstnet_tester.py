from __future__ import division, print_function
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from face_verification.data_loaders import PandoraDatasetTestFixed
from face_verification.losses import accuracy as accuracy_function
from face_verification.losses import loss as loss_function
from face_verification.models import FirstNet
import numpy as np
import os
from face_verification.utilities import normalization, image_processing
import cv2
# import trainer

# # Cluster
# weights_firstnet_pandora = '/homes/spini/3dv/weights_face_verification/firstnet_pandora.pkl'
# dataset_path_pandora_frontal_0 = '/homes/spini/3dv/lists_face_verification/test_angles_frontal_0_1_2.txt'
# dataset_path_pandora_frontal_3 = '/homes/spini/3dv/lists_face_verification/test_angles_frontal_3_4.txt'
# dataset_path_pandora_frontal_a = '/homes/spini/3dv/lists_face_verification/test_angles_frontal_0_1_2_3_4.txt'
# dataset_path_pandora_non_frontal_0 = '/homes/spini/3dv/lists_face_verification/test_angles_non_frontal_0_1_2.txt'
# dataset_path_pandora_non_frontal_3 = '/homes/spini/3dv/lists_face_verification/test_angles_non_frontal_3_4.txt'
# dataset_path_pandora_non_frontal_a = '/homes/spini/3dv/lists_face_verification/test_angles_non_frontal_0_1_2_3_4.txt'
# dataset_path_pandora_0 = '/homes/spini/3dv/lists_face_verification/test_0_1_2.txt'
# dataset_path_pandora_3 = '/homes/spini/3dv/lists_face_verification/test_3_4.txt'
# dataset_path_pandora_a = '/homes/spini/3dv/lists_face_verification/test_0_1_2_3_4.txt'
#
# dataset_path_pandora = '/homes/spini/3dv/lists_face_verification/test_0_1_2_3_4.txt'
# correct_dataset_path = '/homes/fgrazioli/datasets'

# Vangogh
weights_firstnet_biwi = '/projects/FFD/weights_face_verification/firstnet_biwi.pkl'
weights_firstnet_pandora = '/projects/FFD/weights_face_verification/firstnet_pandora.pkl'
dataset_path_pandora_frontal_0 = '/projects/FFD/lists_face_verification/test_angles_frontal_0_1_2.txt'
dataset_path_pandora_frontal_3 = '/projects/FFD/lists_face_verification/test_angles_frontal_3_4.txt'
dataset_path_pandora_frontal_a = '/projects/FFD/lists_face_verification/test_angles_frontal_0_1_2_3_4.txt'
dataset_path_pandora_non_frontal_0 = '/projects/FFD/lists_face_verification/test_angles_non_frontal_0_1_2.txt'
dataset_path_pandora_non_frontal_3 = '/projects/FFD/lists_face_verification/test_angles_non_frontal_3_4.txt'
dataset_path_pandora_non_frontal_a = '/projects/FFD/lists_face_verification/test_angles_non_frontal_0_1_2_3_4.txt'
dataset_path_pandora_0 = '/projects/FFD/lists_face_verification/test_0_1_2.txt'
dataset_path_pandora_3 = '/projects/FFD/lists_face_verification/test_3_4.txt'
dataset_path_pandora_a = '/projects/FFD/lists_face_verification/test_0_1_2_3_4.txt'

dataset_path_pandora = '/projects/FFD/lists_face_verification/test_0_1_2_3_4.txt'
correct_dataset_path = None
dataset_path_biwi = '/projects/face/list_test_biwi/test_biwi.txt'

class FaceTester(object):
    def __init__(self, generator, target_size, dataset_name, verbose=False):
        self.generator = generator
        self.target_size = target_size
        self.verbose = verbose
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        self.dataset_name = dataset_name

        if dataset_name == 'pandora':
            # TODO WARNING!!!!!!!! ONLY 8192 TEST IMAGES!!!!
            self.pandora_dataset_test = PandoraDatasetTestFixed(
                dataset_path=dataset_path_pandora, target_size=target_size, image_type=1, len_subset=8192,
                correct_dataset_path=correct_dataset_path
            )
            self.pandora_data_loader_test = DataLoader(dataset=self.pandora_dataset_test, num_workers=8, batch_size=64,
                                                       shuffle=True, drop_last=False)

            self.pandora_dataset_final_tests = [
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_frontal_0, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_frontal_3, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_frontal_a, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_non_frontal_0, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_non_frontal_3, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_non_frontal_a, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_0, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_3, target_size=target_size, image_type=1, len_subset=-1
                ),
                PandoraDatasetTestFixed(
                    dataset_path=dataset_path_pandora_a, target_size=target_size, image_type=1, len_subset=-1
                ),
            ]

            self.pandora_data_loader_final_tests = [
                DataLoader(dataset=x, num_workers=8, batch_size=64, shuffle=False, drop_last=False)
                for x in self.pandora_dataset_final_tests
            ]

        elif dataset_name == 'BIWI':
            self.biwi_dataset_final_tests = PandoraDatasetTestFixed(
                dataset_path=dataset_path_biwi, target_size=target_size, image_type=1, len_subset=-1,
                dataset_name=self.dataset_name
            )

            self.biwi_data_loader_final_tests = DataLoader(dataset= self.biwi_dataset_final_tests, num_workers=8,
                                                           batch_size=64, shuffle=False, drop_last=False)

        else:
            raise NotImplementedError

        # ToDo Ricordarsi sempre di mettere il modello in eval() !!!!!!!!!!!!
        self.firstnet_model = FirstNet().cuda()
        if self.dataset_name == 'BIWI':
            self.firstnet_model.load_state_dict(
                torch.load(weights_firstnet_biwi, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0'}))
        else:
            self.firstnet_model.load_state_dict(
                torch.load(weights_firstnet_pandora, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0'}))
        self.firstnet_model.eval()

    def test_firstnet(self, epoch):
        self.generator.eval()

        if self.dataset_name == 'BIWI':
            # Biwi
            epoch_loss_biwi = 0
            epoch_accu_biwi = 0
            nof_steps = len(self.biwi_data_loader_final_tests)
            # nof_samples = len(pandora_data_loader_test) * pandora_data_loader_test.batch_size

            for step, (img1, img2, y) in enumerate(self.biwi_data_loader_final_tests):

                img1 = Variable(img1, volatile=True).cuda()
                img2 = Variable(img2, volatile=True).cuda()
                y = Variable(y, volatile=True).float().cuda()

                img1 = self.generator(img1)
                img2 = self.generator(img2)

                img1 = torch.cat([img1, img1, img1], dim=1)
                img2 = torch.cat([img2, img2, img2], dim=1)

                loss_biwi, accu_biwi = self.predict(img1, img2, y)
                epoch_loss_biwi += loss_biwi.data[0]
                epoch_accu_biwi += accu_biwi.data[0]

                if self.verbose:
                    print("\033[1AEpoch %d - Test Face Verification Biwi - Step %4d/%-4d: loss %.6f - acc: %.6f" %
                          (epoch, step, nof_steps, epoch_loss_biwi / (step + 1), epoch_accu_biwi / (step + 1)))
                    # sys.stdout.flush()

            epoch_loss_biwi /= nof_steps
            epoch_accu_biwi /= nof_steps

            # print(epoch_loss_pandora, epoch_accu_pandora)

            if self.verbose:
                print("\033[1AEpoch %d - Test Face Verification Biwi: loss %.6f - acc: %.6f    \033[K" %
                      (epoch, epoch_loss_biwi, epoch_accu_biwi), flush=True)
            else:
                print("Epoch %d - Test Face Verification Biwi: loss %.6f - acc: %.6f" %
                      (epoch, epoch_loss_biwi, epoch_accu_biwi), flush=True)

        elif self.dataset_name == 'pandora':
            # Pandora
            epoch_loss_pandora = 0
            epoch_accu_pandora = 0
            nof_steps = len(self.pandora_data_loader_test)
            # nof_samples = len(pandora_data_loader_test) * pandora_data_loader_test.batch_size

            for step, (img1, img2, y) in enumerate(self.pandora_data_loader_test):

                img1 = Variable(img1, volatile=True).cuda()
                img2 = Variable(img2, volatile=True).cuda()
                y = Variable(y, volatile=True).float().cuda()

                img1 = self.generator(img1)
                img2 = self.generator(img2)

                # trainer.combine_images(img1, 'tristezza1.png')
                # trainer.combine_images(img2, 'tristezza2.png')

                loss_pandora, accu_pandora = self.predict(img1, img2, y)
                epoch_loss_pandora += loss_pandora.data[0]
                epoch_accu_pandora += accu_pandora.data[0]

                if self.verbose:
                    print("\033[1AEpoch %d - Test Face Verification Pandora - Step %4d/%-4d: loss %.6f - acc: %.6f" %
                          (epoch, step, nof_steps, epoch_loss_pandora / (step + 1), epoch_accu_pandora / (step + 1)))
                    # sys.stdout.flush()

            epoch_loss_pandora /= nof_steps
            epoch_accu_pandora /= nof_steps

            # print(epoch_loss_pandora, epoch_accu_pandora)

            if self.verbose:
                print("\033[1AEpoch %d - Test Face Verification Pandora: loss %.6f - acc: %.6f    \033[K" %
                      (epoch, epoch_loss_pandora, epoch_accu_pandora), flush=True)
            else:
                print("Epoch %d - Test Face Verification Pandora: loss %.6f - acc: %.6f" %
                      (epoch, epoch_loss_pandora, epoch_accu_pandora), flush=True)

        else:
            raise NotImplementedError

        self.generator.train()

    def final_test_firstnet(self):
        self.generator.eval()

        if self.dataset_name == 'BIWI':

            data_loader = self.biwi_data_loader_final_tests
            # Biwi
            epoch_loss_biwi = 0
            epoch_accu_biwi = 0
            nof_steps = len(data_loader)

            for step, (img1, img2, y) in enumerate(data_loader):

                img1 = Variable(img1, volatile=True).cuda()
                img2 = Variable(img2, volatile=True).cuda()
                y = Variable(y, volatile=True).float().cuda()

                img1 = self.generator(img1)
                img2 = self.generator(img2)

                loss_biwi, accu_biwi = self.predict(img1, img2, y)
                epoch_loss_biwi += loss_biwi.data[0]
                epoch_accu_biwi += accu_biwi.data[0]

                if self.verbose:
                    print("\033[1AFinal Test Face Verification BIWI %d - Step %4d/%-4d: loss %.6f - acc: %.6f" %
                          (
                          1, step, nof_steps, epoch_loss_biwi / (step + 1), epoch_accu_biwi / (step + 1)))
                    # sys.stdout.flush()

            epoch_loss_biwi /= nof_steps
            epoch_accu_biwi /= nof_steps

            if self.verbose:
                print("\033[1AFinal Test Face Verification BIWI %d: loss %.6f - acc: %.6f    \033[K" %
                      (1, epoch_loss_biwi, epoch_accu_biwi), flush=True)
            else:
                print("Final Test Face Verification Pandora %d: loss %.6f - acc: %.6f" %
                      (1, epoch_loss_biwi, epoch_accu_biwi), flush=True)

        elif self.dataset_name == 'pandora':

            for index, data_loader in enumerate(self.pandora_data_loader_final_tests):
                # Pandora
                epoch_loss_pandora = 0
                epoch_accu_pandora = 0
                nof_steps = len(data_loader)
                # nof_samples = len(pandora_data_loader_test) * pandora_data_loader_test.batch_size

                for step, (img1, img2, y) in enumerate(data_loader):

                    img1 = Variable(img1, volatile=True).cuda()
                    img2 = Variable(img2, volatile=True).cuda()
                    y = Variable(y, volatile=True).float().cuda()

                    img1 = self.generator(img1)
                    img2 = self.generator(img2)

                    # trainer.combine_images(img1, 'tristezza1.png')
                    # trainer.combine_images(img2, 'tristezza2.png')

                    loss_pandora, accu_pandora = self.predict(img1, img2, y)
                    epoch_loss_pandora += loss_pandora.data[0]
                    epoch_accu_pandora += accu_pandora.data[0]

                    if self.verbose:
                        print("\033[1AFinal Test Face Verification Pandora %d - Step %4d/%-4d: loss %.6f - acc: %.6f" %
                              (index, step, nof_steps, epoch_loss_pandora / (step + 1), epoch_accu_pandora / (step + 1)))
                        # sys.stdout.flush()

                epoch_loss_pandora /= nof_steps
                epoch_accu_pandora /= nof_steps

                # print(epoch_loss_pandora, epoch_accu_pandora)

                if self.verbose:
                    print("\033[1AFinal Test Face Verification Pandora %d: loss %.6f - acc: %.6f    \033[K" %
                          (index, epoch_loss_pandora, epoch_accu_pandora), flush=True)
                else:
                    print("Final Test Face Verification Pandora %d: loss %.6f - acc: %.6f" %
                          (index, epoch_loss_pandora, epoch_accu_pandora), flush=True)

        else:
            raise NotImplementedError

        self.generator.train()

    def preproc(self, img1, img2):
        # images for firstnet
        img1 = np.asarray(img1.data, dtype=np.float32)
        img2 = np.asarray(img2.data, dtype=np.float32)

        img1 = (img1 * 117.5) + 117.5
        img2 = (img2 * 117.5) + 117.5

        img1_star = np.ndarray((img1.shape[0], self.target_size, self.target_size))
        img2_star = np.ndarray((img2.shape[0], self.target_size, self.target_size))

        for k in range(len(img1)):
            img1_star[k] = cv2.resize(img1[k][0], (self.target_size, self.target_size))
            img2_star[k] = cv2.resize(img2[k][0], (self.target_size, self.target_size))
            img1_star[k], img2_star[k] = image_processing(img1_star[k], img2_star[k], 0)
            img1_star[k], img2_star[k] = normalization(img1_star[k], img2_star[k], 0)

        img1_star = np.expand_dims(img1_star, 1)
        img2_star = np.expand_dims(img2_star, 1)

        img1_star = Variable(torch.Tensor(img1_star), volatile=True).cuda()
        img2_star = Variable(torch.Tensor(img2_star), volatile=True).cuda()

        return img1_star, img2_star

    def predict(self, img1, img2, y):
        img1, img2 = self.preproc(img1, img2)

        prediction = self.firstnet_model.forward(img1, img2)
        prediction = torch.squeeze(prediction, dim=-1)

        # same_accu1 = torch.mean((torch.round(
        #     torch.squeeze(firstnet_model.forward(img1, img1), dim=-1)) == Variable(
        #     torch.ones(img1.shape[0])).cuda()).float())
        #
        # same_accu2 = torch.mean((torch.round(
        #     torch.squeeze(firstnet_model.forward(img2, img2), dim=-1)) == Variable(
        #     torch.ones(img2.shape[0])).cuda()).float())
        #
        # print(same_accu1, same_accu2)

        loss = self.loss_function(prediction, y)
        accu = self.accuracy_function(prediction, y)

        return loss, accu


# if __name__ == '__main__':
#     tester = FaceTester(generator=None, target_size=96, verbose=False)
#     tester.final_test_firstnet()
