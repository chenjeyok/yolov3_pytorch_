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
    "batch_size": 128,

    "conf_thresh":0.50,
    "nms_thresh":0.20,

    "obj_thresh":0.50,
    "iou_thresh": 0.40,

    "types":{"Person": 0,
             "Driver": 1,
             "Barrel": 2,
             "ICB": 3,
             "ForkLift": 4
             },

    "test_path": "/home/bryce/data/batch6/datasets/coco5/metas/test.txt",
    "img_h": 416,
    "img_w": 416,
    #"parallels": [0],
    "parallels": [0, 2, 3, 4],
    "pretrain_snapshot": "../darknet_53/size960x960_try5/model00.pth",
}
