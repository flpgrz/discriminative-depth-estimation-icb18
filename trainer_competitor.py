from datetime import datetime
from torch import FloatTensor
from torch.optim import Optimizer
from torch import nn
from torch.autograd import Variable
from torch.nn import Module as Model
from torch.utils.data import DataLoader
from avg_meter import AVGMeter
import torch
from torch.nn.modules.loss import _Loss as Loss
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import cv2
from torch.nn import CrossEntropyLoss
from face_verification import firstnet_tester


def combine_images(imgs, path):
    if type(imgs) is Variable:
        imgs = imgs.data.cpu().numpy()
    else:
        imgs = imgs.numpy()
    num = imgs.shape[0]
    channels = imgs.shape[1]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = imgs.shape[2:]
    image = np.zeros((channels, height * shape[0], width * shape[1]), dtype=imgs.dtype)
    for index, img in enumerate(imgs):
        i = int(index / width)
        j = index % width
        for c in range(channels):
            image[c, i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[c, :, :]

    if channels == 1:
        image = (image * 117.5) + 117.5
        image = np.squeeze(image)
    elif channels == 3:
        image = (image * 127.5) + 127.5
        image = image.transpose((1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = image.astype(np.uint8)
    cv2.imwrite(path, image)


class SSELoss(Loss):
    def forward(self, y_pred, y_true):
        return torch.sum((y_pred - y_true) ** 2)


class Trainer(object):
    def __init__(self, exp_path: Path, gen_model: Model, dsc_model: Model,
                 gen_opt: Optimizer, dsc_opt: Optimizer, target_size,
                 test_data_loader: DataLoader, data_loader: DataLoader, dataset_name: str,
                 face_verification_test: bool = False,
                 face_verification_test_from: int = 0,
                 face_verification_test_step: int = 0):

        self.exp_path = exp_path
        self.gen_model = gen_model.cuda()
        self.dsc_model = dsc_model.cuda()
        self.gen_loss = nn.MSELoss().cuda()
        self.dis_loss = CrossEntropyLoss().cuda()
        self.gen_opt = gen_opt
        self.dsc_opt = dsc_opt
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.face_verification_test = face_verification_test
        self.face_verification_test_from = face_verification_test_from
        self.face_verification_test_step = face_verification_test_step
        self.dataset_name = dataset_name
        self.w = 0.001
        self.target_size = target_size
        self.gen_losses = AVGMeter()
        self.dsc_losses = AVGMeter()
        self.tot_losses = AVGMeter()

        checkpoint_path = Path(exp_path, 'last.checkpoint')
        if checkpoint_path.exists():
            print('[loading checkpoint \'{}\']'.format(str(checkpoint_path)))
            self.load_checkpoint(checkpoint_path)
        #     self.epoch = 0
        #     self.start_epoch = 0
        # else:
        #     self.epoch = 0
        #     self.start_epoch = 0
        #     self.test_imgs, self.gt_imgs = self.test_data_loader.__iter__().__next__()
        self.epoch = 0
        self.start_epoch = 0
        self.test_imgs, self.gt_imgs, self.gt_id_label = self.test_data_loader.__iter__().__next__()

        self.step = 0

        #combine_images(self.test_imgs, str(self.exp_path / 'input.png'))
        # img = (img * 117.5) + 117.5

       # combine_images(self.gt_imgs, str(self.exp_path / 'gt.png'))

    # img = (img * 107.9) + 107.9

    def visualize(self):
        self.gen_model.eval()
        output_test = self.gen_model(Variable(self.test_imgs.cuda()))
        combine_images(output_test, str(self.exp_path / '{}.{}.png'.format(self.epoch, self.step)))
        self.gen_model.train()

    def save_checkpoint(self, path: Path) -> bool:
        checkpoint = {
            'gen_model': self.gen_model.state_dict(),
            'dsc_model': self.dsc_model.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'dsc_opt': self.dsc_opt.state_dict(),
            'gen_losses': self.gen_losses,
            'dsc_losses': self.dsc_losses,
            'epoch': self.epoch,
            'test_imgs': self.test_imgs,
            'gt_imgs': self.gt_imgs
        }

        try:
            torch.save(checkpoint, str(path))
            return True
        except:
            return False

    def load_checkpoint(self, path: Path) -> bool:
        try:
            checkpoint = torch.load(str(path))
        except:
            return False

        self.gen_model.load_state_dict(checkpoint['gen_model'])
        self.dsc_model.load_state_dict(checkpoint['dsc_model'])
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        self.dsc_opt.load_state_dict(checkpoint['dsc_opt'])

        self.gen_losses = checkpoint['gen_losses']
        self.dsc_losses = checkpoint['dsc_losses']
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch

        self.test_imgs = checkpoint['test_imgs']
        self.gt_imgs = checkpoint['gt_imgs']

    def fit_stefano(self, input_img, gt_img, gt_id_label):
        # Generator
        input_img = Variable(input_img).cuda()
        gt_img = Variable(gt_img).cuda()
        gt_id_label = Variable(gt_id_label).cuda()

        self.gen_opt.zero_grad()
        self.dsc_opt.zero_grad()

        # Generator
        generated_img = self.gen_model(input_img)
        gen_loss = self.gen_loss(generated_img, gt_img)
        self.gen_losses.append(gen_loss.data[0])

        # Discriminator
        id_prediction = self.dsc_model(generated_img)
        dsc_loss = self.dis_loss(id_prediction, gt_id_label.long())
        self.dsc_losses.append(dsc_loss.data[0])

        # Total loss
        tot_loss = self.w * gen_loss + dsc_loss
        self.tot_losses.append(tot_loss.data[0])

        # Backward
        # Generator backward
        tot_loss.backward(retain_graph=True)
        self.gen_opt.step(closure=None)

        # Discriminator backward
        dsc_loss.backward(retain_graph=True)
        self.dsc_opt.step(closure=None)

    def train(self):
        self.gen_model.train()
        self.dsc_model.train()

        for self.step, (input_img, gt_img, gt_id_label) in enumerate(self.data_loader):

            input_img, gt_img = input_img.cuda(), gt_img.cuda()

            self.fit_stefano(input_img, gt_img, gt_id_label)

            progress = (self.step + 1) / len(self.data_loader)
            progress_bar = ('█' * int(30 * progress)) + ('┈' * (30 - int(30 * progress)))
            if (self.step + 1) % 10 == 0 or progress >= 1:
                print('\r[{}] Epoch {:04d}.{:04d}: ◖{}◗ │ {:6.2f}% │ Dsc_loss: {:.4f} | '
                      'Gen_loss: {:.4f} | Tot_loss: {:.4f}'.format(
                        datetime.now().strftime("%Y-%m-%d@%H:%M"), self.epoch, self.step,
                        progress_bar, 100 * progress,
                        self.dsc_losses.avg,
                        self.gen_losses.avg,
                        self.tot_losses.avg
                      ), end='')

            if self.step % 100 == 0:
                self.visualize()

    def run(self, epochs: int):
        if self.dataset_name == 'biwi':
            dataset = 'BIWI'
        else:
            dataset = 'pandora'


        tester = firstnet_tester.FaceTester(generator=self.gen_model, target_size=self.target_size, verbose=True,
                                            dataset_name=dataset)
        for self.epoch in range(self.start_epoch, epochs):

            self.train()
            self.save_checkpoint(Path(self.exp_path, 'last.checkpoint'))
            if self.face_verification_test \
                    and self.epoch >= self.face_verification_test_from \
                    and (self.epoch - self.face_verification_test_from) % self.face_verification_test_step == 0:
                tester.test_firstnet(self.epoch)

        tester.final_test_firstnet()
