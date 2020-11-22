from config import *
from tqdm import tqdm
import numpy as np


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train_lf(train_loader, model, optimizer, scheduler, config,
             device=device, accumulation_step=ACCULATION):
    model.train()
    summary_loss = AverageMeter()
    for idx, (images, targets) in tqdm(enumerate(train_loader),
                                       total=len(train_loader),
                                       leave=False):
        target_res = {}
        images = torch.stack(images)
        images = images.to(device).float()

        boxes = [torch.from_numpy(target['boxes']).to(device).float() \
                     if type(target['boxes']) is np.ndarray \
                     else target['boxes'].to(device).float() \
                 for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        target_res["img_scale"] = torch.tensor([1.0] * config.batch_size,
                                               dtype=torch.float).to(device)
        target_res["img_size"] = torch.tensor([images[0].shape[-2:]] \
                                              * config.batch_size,
                                              dtype=torch.float).to(device)
        target_res['bbox'] = boxes
        target_res['cls'] = labels
        optimizer.zero_grad()
        outputs = model(images, target_res)
        loss = outputs['loss']
        loss.backward()
        if idx % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        summary_loss.update(loss.detach().item(), images.shape[0])

    return summary_loss.avg


def val_lf(valid_loader, model, config):
    model.eval()
    summary_loss = AverageMeter()
    for steps, (images, targets) in tqdm(enumerate(valid_loader),
                                         total=len(valid_loader),
                                         leave=False):
        with torch.no_grad():
            pred_res = {}
            images = torch.stack(images)
            images = images.to(device).float()

            boxes = [torch.from_numpy(target['boxes']).to(device).float() \
                         if type(target['boxes']) is np.ndarray \
                         else target['boxes'].to(device).float() \
                     for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            pred_res["img_scale"] = torch.tensor([1.0] * config.batch_size,
                                                 dtype=torch.float).to(device)
            pred_res["img_size"] = torch.tensor([images[0].shape[-2:]] \
                                                * config.batch_size,
                                                dtype=torch.float).to(device)
            pred_res['bbox'] = boxes
            pred_res['cls'] = labels

            outputs = model(images, pred_res)
            loss = outputs['loss']

            torch.cuda.synchronize()
            summary_loss.update(loss.detach().item(), images.shape[0])

    return summary_loss.avg