# coding='utf-8'
import torch
import torch.nn as nn
import numpy as np
import math
import logging
from common.utils import bbox_iou


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size # h,w

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[0] / in_h
        stride_w = self.img_size[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # prediction.shape will be [8, 3, 13, 13, 5+5]、[8, 3, 26, 26, 5+5]、[8, 3, 52, 52, 5+5]
        prediction = input.view(bs,  self.num_anchors, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        #logging.debug(prediction.shape)

        # Get outputs
        # x.shape [8,3,18,30]
        # pred_cls.shape [8,3,18,30,5] for 5 classes
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # This is for training, we calculate the losses
        if targets is not None:
            # mask 把总数为(8 x 3 x in_h x in_w)的预测位（Prediction Slot) 从Ground Truth里对应到如果框框内有物体，这个点的值就是1，没有就是0
            mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors, in_w, in_h)
            mask, tx, ty, tw, th = mask.cuda(), tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            #logging.debug(mask.shape)
            tconf, tcls = tconf.cuda(), tcls.cuda()
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask) / 2
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask) / 2
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
            loss_conf = self.bce_loss(conf * mask, mask) + \
                self.lambda_noobj * self.bce_loss(conf * (1 - mask), mask * (1 - mask))
            # pred_cls[mask == 1]这个操作会把pred_cls的空间结构中mask部分缩合为1维，结果例如[41,5]、[26,5]、[38,5]
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                loss_h.item(), loss_conf.item(), loss_cls.item()
        # This is for evaluating, we need the predictions
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_h, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_w, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, anchors, in_w, in_h):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shape
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                #logging.debug('gw:%f anchors[best_n][0]%f' % (gw,anchors[best_n][0]))
                #logging.debug('gh:%f anchors[best_n][1]%f' % (gh,anchors[best_n][1]))
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)

                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, tx, ty, tw, th, tconf, tcls
