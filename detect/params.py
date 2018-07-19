TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 5,
    },
    "batch_size": 1,

    "conf_thresh":0.15,
    "nms_thresh":0.40,

    "obj_thresh":0.50,
    "iou_thresh": 0.40,

    "val_path": "/home/bryce/data/batch6/datasets/coco5/metas/test.txt",
    "img_h": 960,
    "img_w": 960,
    "parallels": [2],
    "pretrain_snapshot": "../darknet_53/size960x960_try5/model66.pth",

    "types":{"Person": 0, "Driver": 1,
             "Barrel": 2,
             "ICB": 3,
             "ForkLift": 4
             }

}
