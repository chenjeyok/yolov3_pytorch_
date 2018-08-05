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

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset
from common.ai_prime_dataset import AIPrimeDataset
from common.utils import non_max_suppression, bbox_iou, class_nms



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
    dataloader = torch.utils.data.DataLoader(dataset=AIPrimeDataset(config["test_path"]),
                                             batch_size=config["batch_size"],
                                             shuffle=False, num_workers=8, pin_memory=False)

    # Start the eval loop
    #logging.info("Start eval.")
    n_gt = 0
    correct = 0
    #logging.debug('%s' % str(dataloader))

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
            output = torch.cat(output_list, dim=1)

            #logging.info('%s' % str(output.shape))

            # 进行NMS抑制
            #output = non_max_suppression(prediction=output, num_classes=config["yolo"]["classes"], conf_thres=config["conf_thresh"], nms_thres=config["nms_thresh"])
            output = class_nms(prediction=output, num_classes=config["yolo"]["classes"],conf_thres=config["conf_thresh"], nms_thres=config["nms_thresh"])
            #  calculate
            for sample_i in range(labels.size(0)):

                # 计算所有的预测数量
                sample_pred = output[sample_i]
                if sample_pred is not None:
                    #logging.debug(sample_pred.shape)
                    for i in range(sample_pred.shape[0]):
                        pred_histro[int(sample_pred[i,6])] +=  1

                # Get labels for sample where width is not zero (dummies)
                target_sample = labels[sample_i, labels[sample_i, :, 3] != 0]
                # Ground truth的 分类编号obj_cls、相对中心x、相对中心y、相对宽w、相对高h
                n_gt=0
                correct=0
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

                    sample_pred = output[sample_i]
                    if sample_pred is not None:
                        # Iterate through predictions where the class predicted is same as gt
                        # 对于每一个ground truth，遍历预测结果
                        for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:  # 如果当前预测分类 == 当前真实分类
                            #logging.info("%d" % obj_cls)
                            box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                            #pred_histro[int(obj_pred)] += 1
                            iou = bbox_iou(box_pred, box_gt)
                            #if iou >= config["iou_thres"] and obj_conf >= config["obj_thresh"]:
                            if iou >= config["iou_thresh"]:
                                correct += 1
                                correct_histro[int(obj_pred)] += 1
                                break
                #logging.debug("----------------")
                #logging.debug(correct_histro[4])
                #logging.debug(pred_histro[4])
                #logging.debug(gt_histro[4])
    if n_gt:
        types = config["types"]

        reverse_types = {}  # 建立一个反向的types
        for key in types.keys():
            reverse_types[types[key]] = key

        #logging.info('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))
        logging.info('Precision:%s' % str([reverse_types[i] +':'+ str(int(100 * correct_histro[i] / pred_histro[i])) for i in range(config["yolo"]["classes"]) ]))
        logging.info('Recall   :%s' % str([reverse_types[i] +':'+ str(int(100 * correct_histro[i] / gt_histro[i])) for i in range(config["yolo"]["classes"])]))

        #logging.info('Mean Average Precision: %.5f' % float(correct / n_gt))


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
    # best 24/25
    # best 31/35
    # best 38/62; best 39/62 after IoU/B1/B2
    for i in range(0, 11):
        #for conf_thresh in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: # m39 best 0.11
        for conf_thresh in [#(0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30),
                            #(0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40),
                            #(0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50),
                            #(0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60),
                            #(0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70),
                            (0.70, 0.50, 0.70, 0.70, 0.40, 0.70, 0.70, 0.60)]:
            #for nms_thresh in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: # m39 best 0.40 on IoU/B1/B2
            for nms_thresh in [(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4)]:
                for iou_thresh in[0.4]:
                # this one the smaller is the better, so there's no need to sweep it
                    logging.info("model%.2d, conf_thresh=%s nms_thresh=%s(IoU/B1/B2) iou_thresh=%.2f" % (i, str(conf_thresh),str(nms_thresh),iou_thresh))
                    config["conf_thresh"] = conf_thresh
                    config["nms_thresh"] = nms_thresh
                    config["test_path"]="/home/bryce/data/batch2/datasets/coco7/metas/valid.txt"
                    config["pretrain_snapshot"]= "../darknet_53/size416x416_try1/model%.2d.pth" % i
                    evaluate(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error('User KeyboardInterrupt, exit')
        exit(0)
