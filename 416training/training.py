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

from tensorboardX import SummaryWriter

# 当前文件所在的路径
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
# 系统 $PATH变量 在0位置插入当前文件所在路径的上层，是为了找到common和nets
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))

# model main是输出预测的主网络，yolo_loss是专门用于训练的计算loss的网络
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset
from common.ai_prime_dataset import  AIPrimeDataset
from common.utils import non_max_suppression, bbox_iou



def train(config):
    # Hyper-parameters
    config["global_step"] = config.get("start_step", 0)
    is_training =  True

    # Net & Loss & Optimizer
    ## Net Main
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    ## YOLO Loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_loss = YOLOLoss(config["yolo"]["anchors"][i],
                             config["yolo"]["classes"], (config["img_w"], config["img_h"]))
        yolo_losses.append(yolo_loss)

    ## Optimizer and LR scheduler
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["lr"]["decay_step"], gamma=config["lr"]["decay_gamma"])

    net = nn.DataParallel(net)
    net = net.cuda()

    # Load checkpoint
    if config["pretrain_snapshot"]:
        logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

    # DataLoader
    dataloader = torch.utils.data.DataLoader(AIPrimeDataset(config["train_path"]),
                                             batch_size=config["batch_size"],
                                             shuffle=True, num_workers=16, pin_memory=True)

    # Start the training
    logging.info("Start training.")
    for epoch in range(config["start_epoch"], config["epochs"]):
        for step, (images, labels) in enumerate(dataloader):
            start_time = time.time()
            config["global_step"] += 1

            # Forward
            outputs = net(images)

            # Loss
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = [[]] * len(losses_name)
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]

            # Zero & Backward & Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f " %
                    (epoch, step, _loss, example_per_second, lr)
                )

        # Things to be done for every epoch
        ## LR schedule
        lr_scheduler.step()
        ## Save checkpoint
        _save_checkpoint(net.state_dict(), config, epoch)

    # Finish training
    logging.info("QiaJiaBa~ BeiBei")

def _save_checkpoint(state_dict, config, epoch, evaluate_func=None):
    # global best_eval_result
    # 从config中读取main里构造的保存目录，再拼接为modle.pth这个名称
    checkpoint_path = os.path.join(config["sub_working_dir"], "model%.2d.pth"%epoch)
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def _get_optimizer(config, net):
    optimizer = None

    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    # 检查参数数量，需要两个参数，一个是training.py 一个是配置params.py
    if len(sys.argv) != 2:
        logging.error("Usage: python training.py params.py")
        sys.exit()
    # 参数2是配置，"params.py"
    params_path = sys.argv[1]
    # 检查该路径下是否存在这个配置文件
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    # 去除后缀名'.py'，用importlib.import_module导入这个模块
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    # mini-batch的size在这里乘上GPU数作为一个Iteration的大Batch_Size
    # 因此，样本总数 =  Batch_Size * GPUs * Iterations
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    # 从配置文件中读取工作目录，并构造成本次训练权重的用的路径
    sub_working_dir = '{}/{}/size{}x{}_try{}'.format(
        config['working_dir'], config['model_params']['backbone_name'],
        config['img_w'], config['img_h'], config['try'])
    # 创建该保存路径路径
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    # 顺便在内存的config字典中保存一下这个保存用的子路径
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    # 顺便实例化一个SummaryWriter，输出的数据可以被tensorboard读取显示出来
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    # 读取配置中使用的GPU list，然后组合为"0,2,3,4"这样的字符串
    # 再设置为系统的环境变量，$CUDA_VISIBLE_DEVICES，
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))

    # 开启训练！！！！
    #config["train_path"] = "/home/bryce/data/batch_all/coco7_train.txt"
    config["train_path"] = "/home/bryce/data/batch2/datasets/coco7/metas/train.txt"
    config["start_epoch"]=8
    config["epochs"] = 50
    config["pretrain_snapshot"]= "/home/bryce/yolov3_pytorch_/darknet_53/size416x416_try0/model%.2d.pth" % (config["start_epoch"]-1)   # load checkpoint
    #config["pretrain_snapshot"]= ""
    train(config)


if __name__ == "__main__":
    main()