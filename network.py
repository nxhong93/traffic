from config import *
import gc
import torch
import torch.nn as nn
from collections import OrderedDict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet


class EfficientDetCus(nn.Module):
    def __init__(self, model, num_class):
        super(EfficientDetCus, self).__init__()

        config = get_efficientdet_config(f'tf_efficientdet_{model}')
        config.num_classes = num_class
        config.image_size = [TRAIN_SIZE, TRAIN_SIZE]
        model = EfficientDet(config=config, pretrained_backbone=False)
        model.class_net = HeadNet(config, num_outputs=config.num_classes)
        self.model = DetBenchTrain(model, config)

    def forward(self, image, target):
        out = self.model(image, target)
        return out



class EfficientDetPred(nn.Module):
    def __init__(self, model_weight, num_class):
        super(EfficientDetPred, self).__init__()

        config = get_efficientdet_config(f'tf_efficientdet_{MODEL_USE}')
        config.num_classes = num_class
        config.image_size = [TRAIN_SIZE, TRAIN_SIZE]
        model = EfficientDet(config, pretrained_backbone=False)
        model.class_net = HeadNet(config, num_outputs=config.num_classes)

        new_keys = model.state_dict().keys()
        values = torch.load(model_weight, map_location=lambda storage, loc: storage).values()
        model.load_state_dict(OrderedDict(zip(new_keys, values)))
        self.model = DetBenchPredict(model)

        del new_keys, values
        gc.collect()

    def forward(self, image, img_info=None):
        out = self.model(image, img_info)
        return out