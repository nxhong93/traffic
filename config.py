import os
import torch
import albumentations as al
from albumentations.pytorch import ToTensorV2, ToTensor
import argparse


parser = argparse.ArgumentParser(description='Training Config', add_help=False)


parser.add_argument('--sub_path', default='./public_sample_submission.json')
parser.add_argument('--train_csv', default='./za_traffic_2020/traffic_train/train_traffic_sign_dataset.json')
parser.add_argument('--train_image', default='./za_traffic_2020/traffic_train/images')
parser.add_argument('--test_image', default='data/')
parser.add_argument('--out_path', default='weight/')
parser.add_argument('--result_folder', default='result/')
parser.add_argument('--save_image', default='save_image/')

args = parser.parse_args()

SEED = 89
FOLD_NUM = 10
TRAIN_SIZE = 1024
PADTH_RATIO = 0.1
MOSAIC_RATIO = 0.4
ACCULATION = 1
USE_APEX = True
MODEL_USE = 'd2'
ORG_SIZE = [626, 1622]


class GlobalConfig:
    num_workers = 12
    batch_size = 1
    n_epochs = 20
    lr = 1e-2
    verbose = 1
    verbose_step = 1
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.2,
        patience=1,
        threshold_mode='abs',
        min_lr=1e-7
    )


class PredictConfig:
    SCORE_THRESH = 0.1
    IOU_THRESH = 0.01
    SKIP_THRESH = 0.0
    SCORE_LAST = 0.05
    IOU_THRESH2=0.1


train_transform = al.Compose([
    al.VerticalFlip(p=0.5),
    al.HorizontalFlip(p=0.5),
    al.RandomGamma(gamma_limit=(50, 150), p=0.5),
    ToTensorV2(p=1)
], bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']})


val_transform = al.Compose([
    ToTensorV2(p=1)
], bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']})


test_transform = al.Compose([
    ToTensorV2(p=1)
], p=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')