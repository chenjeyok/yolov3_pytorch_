TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
        #"backbone_pretrained": "",  # set empty to disable

    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 5,
        "class_names":{"Person": 0, "Driver": 1, "Barrel": 2, "ICB": 3, "ForkLift": 4}

    },
    "lr": {
        "backbone_lr": 0.005,
        "other_lr": 0.005,
        "freeze_backbone": True,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 5,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 128,
    #"validation_batch_size":8,
    #"train_path": "/home/bryce/data/batch13/datasets/coco5_sym/metas/train.txt",
    #"valid_path": "/home/bryce/data/batch13/datasets/coco5/metas/valid.txt",
    #"test_path": "/home/bryce/data/batch13/datasets/coco5/metas/test.txt",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0,2,3,4],                         #  config GPU device
    "working_dir": "/home/bryce/yolov3_pytprch_",    #  replace with your working dir
    #"epochs": 25,
    #"start_epoch":30,
    # pretrain_snapshot is deprecated, it will be changed before loading network
    #"pretrain_snapshot": "/home/bryce/yolov3_pytorch_/darknet_53/size960x960_try5/model24.pth",  # load checkpoint
    #"pretrain_snapshot": "", #  load checkpoint

    "evaluate_type": "",
    "try": 0,
    "export_onnx": False,

    "types":{"Person": 0,
             "Driver": 1,
             "Barrel": 2,
             "ICB": 3,
             "ForkLift": 4
             }

}
