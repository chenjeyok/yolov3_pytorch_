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
        "classes": 80,
    },
    "batch_size": 32,
    "iou_thres": 0.5,
    "val_path": "/home/bryce/data/prime_49k/test.txt",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0,2,3,4],
    "pretrain_snapshot": "../darknet_53/size416x416_try0/20180617045156/model.pth",
}
