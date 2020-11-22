from torch.utils.data import Dataset, DataLoader, sampler
from config import *
import random
import numpy as np
import pandas as pd
import cv2
from utils import label_resize



class TrafficDataset(Dataset):
    def __init__(self, df, image_path,
                 img_size=(TRAIN_SIZE, TRAIN_SIZE),
                 transform=None, is_train=True):
        super(TrafficDataset, self).__init__()
        self.df = df
        self.image_path = image_path
        self.list_img = list(df['image_id'].unique())
        self.img_size = img_size
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.list_img)

    def load_image_and_boxes(self, index):
        image_id = self.list_img[index]
        if os.path.exists(f'{self.image_path[0]}/{image_id}.png'):
            image = cv2.imread(f'{self.image_path[0]}/{image_id}.png', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(f'{self.image_path[1]}/{image_id}.png', cv2.IMREAD_COLOR)
        org_size = image.shape[:2]
        image = cv2.resize(image, self.img_size[::-1], cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.is_train:
            records = self.df[self.df['image_id'] == image_id]
            boxes = records[['x', 'y', 'w', 'h', 'category_id']].values
            for box_id in range(len(boxes)):
                boxes[box_id, :-1] = label_resize(org_size, self.img_size, boxes[box_id, :-1])
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            return image, boxes
        else:
            return image, None

    def load_cutmix_image_and_boxes(self, index, padth_ratio=PADTH_RATIO):
        s = self.img_size[0] // 2
        h, w = self.img_size

        xc, yc = [int(random.uniform((0.5 - padth_ratio) * self.img_size[0],
                                     (0.5 + padth_ratio) * self.img_size[0])) for _ in range(2)]
        indexes = [index] + [random.randint(0, len(self.list_img) - 1) for _ in range(3)]
        result_image = np.full((w, h, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            # boxes
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh
            result_boxes.append(boxes)
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, :-1], 0, 2 * s, out=result_boxes[:, :-1])
            result_boxes = result_boxes.astype(np.int32)
            result_boxes = result_boxes[np.where((result_boxes[:, 2] - result_boxes[:, 0]) * \
                                                 (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]

        return result_image, result_boxes

    def __getitem__(self, index):
        img_idx = self.list_img[index]
        if not self.is_train or random.random() > MOSAIC_RATIO:
            image, boxes = self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_cutmix_image_and_boxes(index)
        if self.is_train:
            target = {}
            target['boxes'] = boxes[:, :-1]
            target['labels'] = torch.from_numpy(boxes[:, -1])
            target['img_scale'] = torch.tensor([1.])
            target['image_id'] = torch.tensor([index])
            target['img_size'] = torch.tensor([self.img_size])
            if self.transform is not None:
                for i in range(10):
                    sample = {
                        'image': image,
                        'bboxes': target['boxes'],
                        'labels': target['labels']
                    }
                    sample = self.transform(**sample)
                    if len(sample['bboxes']) > 0:
                        image = sample['image']
                        target['boxes'] = torch.stack(tuple(map(torch.tensor,
                                                                zip(*sample['bboxes'])))).permute(1, 0)
                        target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]
                        break
            if type(image) is np.ndarray:
                image = torch.from_numpy(image).permute(2, 0, 1)
            return image, target
        else:
            if self.transform is not None:
                sample = {'image': image}
                sample = self.transform(**sample)
                image = sample['image']

            img_info = {}
            img_info['img_scale'] = torch.tensor([1.])
            img_info['img_size'] = torch.tensor([self.img_size])
            return image, img_idx, img_info

    def collate_fn(self, batch):
        return tuple(zip(*batch))