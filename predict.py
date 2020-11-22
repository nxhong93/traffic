import pandas as pd
import json
import gc
from config import *
from transform_data import *
from ensemble_boxes import weighted_boxes_fusion, nms
from config import *
from dataset import TrafficDataset
from network import EfficientDetPred
from glob import glob
from IPython.display import display
from torch.utils.data import Dataset, DataLoader, sampler
from tqdm import tqdm
from itertools import product
from utils import *


def run_wbf(predictions, image_index,
            image_size=TRAIN_SIZE,
            iou_thr=PredictConfig.IOU_THRESH,
            iou_thr2=PredictConfig.IOU_THRESH2,
            skip_box_thr=PredictConfig.SKIP_THRESH,
            weights=None):
    boxes = [(prediction[image_index]['boxes'] / (image_size - 1)).tolist() \
             for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [prediction[image_index]['labels'].tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels,
                                                  weights=weights,
                                                  iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)

    new_labels = [1.] * len(labels)
    new_boxes, new_scores, new_labels = weighted_boxes_fusion([boxes], [scores],
                                                              [new_labels],
                                                              weights=weights,
                                                              iou_thr=iou_thr2,
                                                              skip_box_thr=skip_box_thr)
    index_boxes_save = []
    for id_box, new_box in enumerate(new_boxes):
        for box in boxes:
            iou = bb_intersection_over_union(box, new_box)
            if iou > 0.2:
                index_boxes_save.append(id_box)
                break
    scores = [scores[i] for i in index_boxes_save]
    labels = [labels[i] for i in index_boxes_save]
    new_boxes = new_boxes * (image_size - 1)
    return new_boxes, scores, labels


def predict(images, img_info, score_thres=PredictConfig.SCORE_THRESH):
    list_weight = sorted([i for i in os.listdir(args.out_path) if '.pth' in i])
    prediction = []
    for weight in list_weight:
        weight = os.path.join(args.out_path, weight)
        net = EfficientDetPred(model_weight=weight, num_class=len(categorie_df)).to(device)
        net.eval()
        with torch.no_grad():
            for tta_transform in tta_transforms:
                result = []
                det = net(tta_transform.batch_augment(images.clone()), img_info=None)
                for i in range(images.shape[0]):
                    boxes = det[i].detach().cpu().numpy()[:, :4]
                    scores = det[i].detach().cpu().numpy()[:, 4]
                    labels = det[i].detach().cpu().numpy()[:, 5]
                    indexes = np.where(scores > score_thres)[0]
                    boxes = boxes[indexes]
                    labels = labels[indexes]
                    boxes = tta_transform.deaugment_boxes(boxes.copy())
                    result.append({
                        'labels': labels[indexes],
                        'boxes': boxes,
                        'scores': scores[indexes],
                    })

                prediction.append(result)

    return prediction


def wbf_cal(test_loader):
    all_results = []
    for images, image_ids, img_info in tqdm(test_loader, leave=False,
                                            total=len(test_loader)):
        images = torch.stack(images).to(device).float()
        predictions = predict(images, img_info=None)
        for i, (image, img_id) in enumerate(zip(images, image_ids)):
            boxes, scores, labels = run_wbf(predictions, image_index=i)
            boxes = (boxes * 1).round(2).clip(min=0, max=TRAIN_SIZE - 1)
            image_id = image_ids[i]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            results = []
            for box, score, label in zip(boxes, scores, labels):
                if score > PredictConfig.SCORE_LAST:
                    box = label_resize([TRAIN_SIZE, TRAIN_SIZE], ORG_SIZE, box)
                    result = dict()
                    result["image_id"] = image_id
                    result["category_id"] = int(label)
                    result["bbox"] = [i.round(2) for i in box]
                    result["score"] = score
                    results.append(result)
                else:
                    break
            # print(f'{image_id} have {len(results)} bounding box!')
            all_results.extend(results)
    print(50 * '-')
    print(f'Total have {len(all_results)} bounding box!')
    return all_results


if __name__=='__main__':
    # read file
    with open(args.train_csv, 'r') as f:
        train_json = json.loads(f.read())
    # List label
    categorie_df = pd.DataFrame.from_records(train_json['categories'])
    display((categorie_df))

    # Read sample submission
    sub_df = pd.read_json(args.sub_path)

    # image folder
    test_image_path = glob(f'{args.test_image}/*.*')

    # test loader
    list_img_test = sorted([int(i.split('/')[-1][:-4]) for i in test_image_path])
    test_df = pd.DataFrame({'image_id': list_img_test})
    display(test_df.head())
    test_dataset = TrafficDataset(test_df, [args.train_image, args.test_image],
                                  transform=test_transform, is_train=False)
    test_ld = DataLoader(test_dataset, batch_size=8, shuffle=False,
                         collate_fn=test_dataset.collate_fn)
    print(f'Number of test dataloader: {len(test_ld)}')

    # TTA Transform
    tta_transforms = []
    for tta_combination in product([TTAHorizontalFlip(), None],
                                   [TTAVerticalFlip(), None]):
        tta_transforms.append(TTACompose([tta_transform \
                                          for tta_transform in tta_combination \
                                          if tta_transform]))

    # Predict process
    predictions = wbf_cal(test_ld)
    os.makedirs(args.result_folder, exist_ok=True)
    save_json(predictions, f'{args.result_folder}/submission.json')

    display(display_output(test_df, categorie_df, predictions, args.test_image, num_image=None))