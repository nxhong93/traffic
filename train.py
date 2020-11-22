from config import *
from utils import split_df, label_resize
from engineer import train_lf, val_lf
import json
import gc
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from network import EfficientDetCus
from dataset import TrafficDataset
from torch.utils.data import Dataset, DataLoader, sampler
from IPython.display import display
from glob import glob


class Train_process(object):
    def __init__(self, device=device, config=GlobalConfig):
        super(Train_process, self).__init__()
        self.config = config
        self.device = device

    def process_data(self, train_df, split_df, fold_idx):
        img_train = split_df[split_df[f'fold_{fold_idx}'] == 0]['image_id'].tolist()
        train_data = train_df[train_df['image_id'].isin(img_train)].reset_index(drop=True)
        val_data = train_df[~train_df['image_id'].isin(img_train)].reset_index(drop=True)
        # Create dataset
        train_dataset = TrafficDataset(train_data, [args.train_image, args.test_image],
                                       transform=train_transform, is_train=True)
        valid_dataset = TrafficDataset(val_data, [args.train_image, args.test_image],
                                       transform=val_transform, is_train=True)
        # Create dataloader
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=True, pin_memory=True,
                                  num_workers=self.config.num_workers)
        valid_loader = DataLoader(valid_dataset, pin_memory=True,
                                  batch_size=self.config.batch_size,
                                  collate_fn=valid_dataset.collate_fn,
                                  num_workers=self.config.num_workers)

        del img_train, train_data, val_data, train_dataset, valid_dataset
        gc.collect()
        return train_loader, valid_loader

    def fit(self, train_df, split_df):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        for fold in range(FOLD_NUM // 2)[:-1]:
            print(50 * '-')
            print(f'Fold{2 * fold}:')
            model = EfficientDetCus(model=MODEL_USE, num_class=len(categorie_df)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
            scheduler = self.config.SchedulerClass(optimizer, **self.config.scheduler_params)
            train_loader, valid_loader = self.process_data(train_df, split_df, 2 * fold)

            best_val_loss = np.Inf
            for epoch in range(self.config.n_epochs):
                train_loss = train_lf(train_loader, model, optimizer,
                                      scheduler, config=self.config)
                val_loss = val_lf(valid_loader, model, config=self.config)
                print(f'Epoch{epoch}: \tTrain_loss: {train_loss:.5f} | Val_loss: {val_loss:.5f}')

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    # torch.save(model.state_dict(), f'{args.out_path}model_fold{fold}.pth')
                    print('Model improved, saving model!')

                if self.config.validation_scheduler:
                    scheduler.step(val_loss)
            torch.cuda.empty_cache()


if __name__=='__main__':
    # read file
    with open(args.train_csv, 'r') as f:
        train_json = json.loads(f.read())
    train_image_df = pd.DataFrame.from_records(train_json['images'])
    heigh_width_unique = list(set(zip(*map(train_image_df.get, ['height', 'width']))))
    print(f'The list different size image: {heigh_width_unique}')

    # Train file
    annot_df = pd.DataFrame.from_records(train_json['annotations'])
    annot_df['weight'] = heigh_width_unique[0][1]
    annot_df['height'] = heigh_width_unique[0][0]
    annot_df['x'] = annot_df['bbox'].apply(lambda x: x[0])
    annot_df['y'] = annot_df['bbox'].apply(lambda x: x[1])
    annot_df['w'] = annot_df['bbox'].apply(lambda x: x[2])
    annot_df['h'] = annot_df['bbox'].apply(lambda x: x[3])
    display(annot_df.head())

    # List label
    categorie_df = pd.DataFrame.from_records(train_json['categories'])
    display((categorie_df))

    # List image in folder
    train_image_path = glob(f'{args.train_image}/*.*')
    test_image_path = glob(f'{args.test_image}/*.*')

    print(f'Number of train image: {len(train_image_path)}, test image: {len(test_image_path)}')

    # Create fold
    annot_pivot = split_df(annot_df)
    display(annot_pivot.head())

    #Train process
    train_pr = Train_process()
    train_pr.fit(annot_df, annot_pivot)