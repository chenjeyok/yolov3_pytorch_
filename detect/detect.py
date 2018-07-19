# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset

from common.utils import non_max_suppression, bbox_iou, draw_prediction
import cv2


def evaluate(config):
    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        logging.warning("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader
    dataloader = torch.utils.data.DataLoader(dataset=COCODataset(config["val_path"], config["img_w"]),
                                             batch_size=config["batch_size"],
                                             shuffle=True, num_workers=1, pin_memory=False)

    # Start the eval loop
    logging.info("Start eval.")
    n_gt = 0
    correct = 0
    logging.info('%s' % str(dataloader))

    gt_histro={}
    pred_histro = {}
    correct_histro = {}

    for i in range(config["yolo"]["classes"]):
        gt_histro[i] = 1
        pred_histro[i] = 1
        correct_histro[i] = 0

    # images 是一个batch里的全部图片，labels是一个batch里面的全部标签
    for step, (images, labels) in enumerate(dataloader):
        labels = labels.cuda()
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))

            # 把三个尺度上的预测结果在第1维度(第0维度是batch里的照片，第1维度是一张照片里面的各个预测框，第2维度是各个预测数值)上拼接起来
            batch_output = torch.cat(output_list, dim=1)

            logging.info('%s' % str(batch_output.shape))

            # 进行NMS抑制
            batch_output = non_max_suppression(prediction=batch_output, num_classes=config["yolo"]["classes"], conf_thres=config["conf_thresh"], nms_thres=config["nms_thresh"])
            #  calculate
            for sample_index_in_batch in range(labels.size(0)):
                # fetched img sample in tensor( C(RxGxB) x H x W ), transform to cv2 format in  H x W x C(BxGxR)
                sample_image = images[sample_index_in_batch].numpy()
                sample_image = np.transpose(sample_image, (1, 2, 0))
                sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

                logging.debug("fetched img %d size %s" % (sample_index_in_batch, sample_image.shape))
                # Get labels for sample where width is not zero (dummies)(init all labels to zeros in array)
                target_sample = labels[sample_index_in_batch, labels[sample_index_in_batch, :, 3] != 0]
                # get prediction for this sample
                sample_pred = batch_output[sample_index_in_batch]
                if sample_pred is not None:
                    for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred:  # for each prediction box
                        # logging.info("%d" % obj_cls)
                        box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                        sample_image = draw_prediction(sample_image,conf, obj_conf, int(obj_pred), (x1, y1, x2, y2), config)

                # 每一个ground truth的 分类编号obj_cls、相对中心x、相对中心y、相对宽w、相对高h
                for obj_cls, tx, ty, tw, th in target_sample:
                    # Get rescaled gt coordinates
                    # 转化为输入像素尺寸的 左上角像素tx1 ty1，右下角像素tx2 ty2
                    tx1, tx2 = config["img_w"] * (tx - tw / 2), config["img_w"] * (tx + tw / 2)
                    ty1, ty2 = config["img_h"] * (ty - th / 2), config["img_h"] * (ty + th / 2)
                    # 计算ground truth数量，用于统计信息
                    n_gt += 1
                    gt_histro[int(obj_cls)] += 1
                    # 转化为 shape(1,4)的tensor，用来计算IoU
                    box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1)
                    # logging.info('%s' % str(box_gt.shape))

                    sample_pred = batch_output[sample_index_in_batch]
                    if sample_pred is not None:
                        # Iterate through predictions where the class predicted is same as gt
                        # 对于每一个ground truth，遍历预测结果
                        for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:  # 如果当前预测分类 == 当前真实分类
                            #logging.info("%d" % obj_cls)
                            box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                            pred_histro[int(obj_pred)] += 1
                            iou = bbox_iou(box_pred, box_gt)
                            if iou >= config["iou_thresh"]:
                                correct += 1
                                correct_histro[int(obj_pred)] += 1
                                break
        if n_gt:
            types = config["types"]
            reverse_types = {}  # 建立一个反向的types
            for key in types.keys():
                reverse_types[types[key]] = key

            logging.info('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))
            logging.info('mAP Histro:%s' % str([  reverse_types[i] +':'+ str(int(100 * correct_histro[i] / gt_histro[i])) for i in range(config["yolo"]["classes"] )  ]))
            logging.info('Recall His:%s' % str([  reverse_types[i] +':'+ str(int(100 * correct_histro[i] / pred_histro[i])) for i in range(config["yolo"]["classes"]) ]))

    logging.info('Mean Average Precision: %.5f' % float(correct / n_gt))


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python training.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    evaluate(config)


if __name__ == "__main__":

    graphviz = GraphvizOutput(output_file=r'./trace_%s.png' % str(__file__))
    with PyCallGraph(output=graphviz):
        try:
            main()
        except KeyboardInterrupt:
            logging.error('User KeyboardInterrupt, exit')
            exit(0)
