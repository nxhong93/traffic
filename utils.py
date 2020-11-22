import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from config import *
from IPython.display import display
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(pred, json_file):
    with open(json_file, 'w+') as f:
        f.write(json.dumps(pred, cls=NpEncoder, indent=4))


def read_json(json_file):
    with open(json_file, 'r+') as f:
        content = json.load(f)
    return content


def label_resize(org_size, img_size, bbox) -> object:
    x, y, w, h = bbox
    x_new = (x * img_size[1] / org_size[1])
    y_new = (y * img_size[0] / org_size[0])
    w_new = (w * img_size[1] / org_size[1])
    h_new = (h * img_size[0] / org_size[0])
    return x_new, y_new, w_new, h_new




def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def split_df(df):
    kf = MultilabelStratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)
    annot_pivot = pd.pivot_table(df, index=['image_id'], columns=['category_id'],
                                 values='id', fill_value=0, aggfunc='count') \
        .reset_index().rename_axis(None, axis=1)
    for fold, (train_idx, val_idx) in enumerate(kf.split(annot_pivot,
                                                         annot_pivot.iloc[:, 1:8])):
        annot_pivot[f'fold_{fold}'] = 0
        annot_pivot.loc[val_idx, f'fold_{fold}'] = 1
    return annot_pivot


def ListColor(df):
    class_unique = sorted(df['id'].unique().tolist())
    dict_color = dict()
    for classid in class_unique:
        dict_color[classid] = random.sample(range(256), 3)

    return dict_color


def display_output(df, list_df, json, folder, num_image=1):
    dict_color = ListColor(list_df)
    if num_image is None:
        list_image = df['image_id'].unique()
    else:
        list_image = np.random.choice(df['image_id'].unique(), num_image)
    for img in list_image:
        fig = plt.figure(figsize=(10, 12))
        image_path = os.path.join(folder, f'{img}.png')
        images = cv2.imread(image_path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        list_pred = [i for i in json if i['image_id'] == img]
        for pred in list_pred:
            score = pred['score']
            box = pred['bbox']
            category = pred['category_id']
            x, y, w, h = list(map(int, box))
            cv2.rectangle(images, (x, y), (x + w, y + h), dict_color[category], 2)
            cv2.putText(images, f'{category}: {score:.2f}', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), 2)
        plt.title(f'{img}.png has {len(list_pred)} box', color='w', fontsize=18)
        display(plt.imshow(images))

        os.makedirs(args.save_image, exist_ok=True)
        fig.savefig(f'./{args.save_image}/{img}.png', bbox_inches='tight', dpi=200)
