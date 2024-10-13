import os
import sys

import numpy as np
from PIL import Image

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

def read_masks(path):
    mask = Image.open(path)
    mask = np.array(mask)

    return mask

def calculate_f1_score(gt_mask, pred_mask, num_classes=19):
    # Flatten the masks for F1-score calculation
    gt_mask_flat = gt_mask.flatten()
    pred_mask_flat = pred_mask.flatten()

    # Calculate Precision, Recall, and F1-score for each class
    precision = []
    recall = []
    f1 = []

    for cls_idx in range(num_classes):
        precision.append(precision_score(gt_mask_flat == cls_idx, pred_mask_flat == cls_idx, average='binary', zero_division=0))
        recall.append(recall_score(gt_mask_flat == cls_idx, pred_mask_flat == cls_idx, average='binary', zero_division=0))
        f1.append(f1_score(gt_mask_flat == cls_idx, pred_mask_flat == cls_idx, average='binary', zero_division=0))

    return precision, recall, f1


# replace submit_dir to your result path here
submit_dir = 'test_results'

# replace truth_dir to ground-truth path here
truth_dir = 'Data_preprocessing/val_mask'

# replace output_dir to the desired output path, and you will find 'scores.txt' containing the calcuated mIoU
output_dir = '.'

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit_dir_list = os.listdir(submit_dir)
    if len(submit_dir_list) == 1:
        submit_dir = os.path.join(submit_dir, "%s" % submit_dir_list[0])
        assert os.path.isdir(submit_dir)
        
    area_intersect_all = np.zeros(19)
    area_union_all = np.zeros(19)

    precision_all = np.zeros(19)
    recall_all = np.zeros(19)
    f1_all = np.zeros(19)
    
    # for idx in range(500):
    for idx in tqdm(range(1000)):
        pred_mask = read_masks(os.path.join(submit_dir, "%s.png" % idx))
        gt_mask = read_masks(os.path.join(truth_dir, "%s.png" % idx))
        for cls_idx in range(19):
            area_intersect = np.sum(
                (pred_mask == gt_mask) * (pred_mask == cls_idx))

            area_pred_label = np.sum(pred_mask == cls_idx)
            area_gt_label = np.sum(gt_mask == cls_idx)
            area_union = area_pred_label + area_gt_label - area_intersect

            area_intersect_all[cls_idx] += area_intersect
            area_union_all[cls_idx] += area_union
            
        # Calculate precision, recall, f1 for this image
        precision, recall, f1 = calculate_f1_score(gt_mask, pred_mask)
        precision_all += precision
        recall_all += recall
        f1_all += f1

    iou_all = area_intersect_all / area_union_all * 100.0
    miou = iou_all.mean()
    
    # Average F1-scores
    precision_avg = np.mean(precision_all / 500)
    recall_avg = np.mean(recall_all / 500)
    f1_avg = np.mean(f1_all / 500)

    # Create the evaluation score path
    output_filename = os.path.join(output_dir, 'scores.txt')

    with open(output_filename, 'w') as f3:
        f3.write(f'mIOU: {miou}\n')
        f3.write(f'Precision: {precision_avg}\n')
        f3.write(f'Recall: {recall_avg}\n')
        f3.write(f'F1-Score: {f1_avg}\n')