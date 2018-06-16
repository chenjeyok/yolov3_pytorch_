TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 1,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-04,
    },
    "batch_size": 32,
    "train_path": "/home/bryce/data/prime_49k/train.txt",
    "epochs": 10,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0,2,3,4],                         #  config GPU device
    "working_dir": "/home/bryce/YOLOv3_PyTorch",    #  replace with your working dir
    "pretrain_snapshot": "../darknet_53/size416x416_try0/20180617045156/model.pth",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,
}
