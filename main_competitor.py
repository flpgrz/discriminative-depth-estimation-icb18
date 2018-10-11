import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

import click
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer_competitor import Trainer
from models_competitor import Generator, Discriminator
from click import Path as ClickPath
from datasets_competitor import Faces
from pathlib import Path
import numpy as np
import random

SEED = 1821
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    # 	m.weight.data.normal_(1.0, 0.02)
    # 	m.bias.data.fill_(0)


@click.command()
@click.option('--exp_name', type=str, default='competitor_paper_96_pandora ')  # , prompt='Enter --exp_name'
@click.option('--dataset_dir_path', type=ClickPath(exists=True), default="/projects/GAN-FFD/dataset")  # "/projects/GAN-FFD/dataset" OR /projects/3DVISION/BIWI_ok
@click.option('--dataset_test_folder', type=bool, default=True)  # if True: face_dataset for train and face_dataset_test for test
@click.option('--epochs', type=int, default=100)
@click.option('--batch_size', type=int, default=64)
@click.option('--lr', type=float, default=0.0002)
@click.option('--log_dir_path', type=ClickPath(exists=False), default='log')
@click.option('--dataset_name', type=str, default='pandora' )
@click.option('--target_size', type=int, default=96)
@click.option('--face_verification', type=bool, default=True)
@click.option('--face_verification_from', type=int, default=1)
@click.option('--face_verification_step', type=int, default=1)
def main(exp_name: str, dataset_dir_path: str, dataset_test_folder: bool, epochs: int,
         batch_size: int, lr: float, log_dir_path: str, dataset_name: str,
         target_size: int, face_verification: bool, face_verification_from: int, face_verification_step: int):
    print('Starting Experiment \'{}\''.format(exp_name))

    exp_path = Path(log_dir_path, exp_name)
    if not exp_path.exists():
        exp_path.mkdir(parents=True)

    ds_train = Faces(dataset_dir_path, dataset_name, dataset_test_folder=dataset_test_folder,
                                training=True, target_size=(target_size, target_size))
    ds_test = Faces(dataset_dir_path, dataset_name, dataset_test_folder=dataset_test_folder,
                                training=False, target_size=(target_size, target_size))

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=8)

    gen = Generator().cuda()
    dsc = Discriminator(dataset_name, target_size).cuda()

    gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    dsc_opt = optim.Adam(dsc.parameters(), lr=lr, betas=(0.5, 0.999))

    trainer = Trainer(
        exp_path=exp_path,
        gen_model=gen,
        dsc_model=dsc,
        gen_opt=gen_opt,
        dsc_opt=dsc_opt,
        face_verification_test=face_verification,
        face_verification_test_from=face_verification_from,
        face_verification_test_step=face_verification_step,
        dataset_name=dataset_name,
        data_loader=train_loader,
        test_data_loader=test_loader,
        target_size=target_size
    )

    trainer.run(epochs)


if __name__ == '__main__':
    main()
