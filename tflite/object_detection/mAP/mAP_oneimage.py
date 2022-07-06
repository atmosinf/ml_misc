import sys
import json
import os
from urllib.request import ProxyBasicAuthHandler
import cv2
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image
import tensorflow as tf
import numpy as np


ANNOTATIONS = '../my_custom_dataset/annotations/coco_annotation.json'
labels = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','street sign','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','hat','backpack','umbrella','shoe','eye glasses','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','plate','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','mirror','dining table','window','desk','toilet','door','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','toothbrush','hair brush']

with open(ANNOTATIONS, 'r') as f:
    data = json.load(f)


def get_prediction(model, img_root, img_name):
    filepath = os.path.join(img_root, img_name)
    image = Image.open(filepath)
    imagetensor = tf.convert_to_tensor(image, dtype=tf.uint8, dtype_hint=None, name=None)
    imagetensor_reshaped = tf.expand_dims(imagetensor, axis=0, name=None)
    boxes, scores, classes, num_detections = model(imagetensor_reshaped)
    return boxes, scores, classes, num_detections

def get_gt(img_name):
    img_id = int(img_name.replace('.jpg',''))
    annotations = data['annotations']
    gt = []
    for annot in annotations:
        if annot['image_id'] == img_id:
            gt.append([annot['bbox'], annot['category_id']])
    return gt

def get_gtruths_preds_frmtd(model, img_root, img_name, conf_thresh=0.5, labelidxs=[0,21,23,24]):
    '''
    given an input image, get the gtruths, predictions and return the bboxes for both
    returned bboxes are in the format xmin, ymin, xmax, ymax
    '''
    filepath = os.path.join(img_root, img_name)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get ground truth bboxes
    gt = get_gt(img_name)

    # get predictions
    boxes, scores, classes, num_detections = get_prediction(model, img_root, img_name)
    
    # convert the bboxes to the format xmin, ymin, xmax, ymax
    preds = []
    conf_thresh = 0.5
    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        if score > conf_thresh and int(cls.numpy())-1 in labelidxs:
            ymin, xmin, ymax, xmax = box.numpy()
            preds.append([1, int(cls.numpy()), score.numpy(), xmin, ymin, xmax, ymax])

    gtruths = []
    for gti in gt:
        xmin, ymin, width, height = gti[0]
        cls = gti[1]
        if cls-1 in labelidxs:
            gtruths.append([1, cls, 1, xmin, ymin, xmin+width, ymin+height])
    
    return gtruths, preds

def get_iou(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

import torch
from collections import Counter

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = get_iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
#                     box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / (len(average_precisions) + epsilon)

def viz_gt_and_pred(model, img_root, img_name, conf_threshold=0.5, labelidxs=[0,21,23,24]):
    filepath = os.path.join(img_root, img_name)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # get ground truth bboxes
    gt = get_gt(img_name)
    for gti in gt:
        gt_bbox, category = gti
        if category-1 in labelidxs:
            xmin, ymin, width, height = map(int, gt_bbox)
            cv2.putText(img, labels[category-1], (xmin, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.rectangle(img, (xmin, ymin), (xmin+width, ymin+height), (0, 255, 0), 1)
        
    # get predictions
    boxes, scores, classes, num_detections = get_prediction(model, img_root, img_name)
    for i, score in enumerate(scores[0]):
        if score > conf_threshold:
            index = i
            labelindex = int(classes[0][index].numpy()) - 1
            if labelindex in labelidxs:
                ymin, xmin, ymax, xmax = map(int, boxes[0][index].numpy())
                cv2.putText(img, labels[labelindex], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                cvrect = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
    
    plt.figure(figsize=(10,7))
    plt.imshow(img)
    plt.show()


def run():
    img_name = sys.argv[1]
        
    labelidxs=[0,21,23,24]
    model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")
    img_root = '../my_custom_dataset/images/'
    conf_threshold = 0.5
    iou_threshold = 0.7
    num_classes = len(labels)
    

    gtruths, preds = get_gtruths_preds_frmtd(model, img_root, img_name, conf_threshold, labelidxs)
    mean_avg_prec = mean_average_precision(preds, gtruths, iou_threshold, box_format='corner', num_classes=num_classes)
    print(f'Precision={mean_avg_prec:.4f} @iou_threshold={iou_threshold}')

    viz_gt_and_pred(model, img_root, img_name, 0.5)

run()

'''
to run this file, use this format: 
python mAP_oneimage.py IMAGENAME.jpg
change the parameters such as image_root, image_name, confidence_threshold etc in the run() function
'''