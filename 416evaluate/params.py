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
        "classes": 8,
    },
    "batch_size": 128,

    "iou_thresh": 0.40,

    "types": {"Head":0, "Cloth":1 ,"Person": 2, "Driver": 3, "Barrel": 4, "ICB": 5, "ForkLift_1": 6, "ForkLift_2":7},

    "test_path": "/home/bryce/data/batch6/datasets/coco7_sym/metas/valid.txt",
    "img_h": 416,
    "img_w": 416,
    #"parallels": [0],
    "parallels": [0,2,3,4],
    "pretrain_snapshot": "../darknet_53/size960x960_try5/model00.pth",
}
