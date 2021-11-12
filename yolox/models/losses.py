#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import torch
import torch.nn as nn

from shapely.geometry import Polygon,MultiPoint, polygon #多边形


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class PolyIOUloss(nn.Module):
    """
        For Caculating iou loss between two polygons
    """
    def __init__(self, reduction="none", loss_type="iou"):
        super(PolyIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, preds, targets):
        ious = []
        if self.loss_type == "giou":
            gious = []
        for pred, target in zip(preds, targets):
            pred = pred.reshape(4,2)
            target = target.reshape(4,2)

            pred_poly = Polygon(pred).convex_hull
            target_poly = Polygon(target).convex_hull
            union_poly = Polygon(torch.cat((pred,target))).convex_hull

            if self.loss_type == "iou":
                if not pred_poly.intersects(target_poly):
                    iou = 0;
                else:
                    iou = pred_poly.intersection(target_poly).area / union_poly.area
                ious.append(iou)

            elif self.loss_type == "giou":
                if not pred_poly.intersects(target_poly):
                    iou = 0
                    giou = -1
                else:
                    iou = pred_poly.intersection(target_poly).area / union_poly.area
                    giou = iou - ((union_poly.area - pred_poly.intersection(target_poly).area) / union_poly.area)
                
                ious.append(iou)
                gious.append(giou)

        ious = torch.tensor(ious)
        if self.loss_type == "giou":
            gious = torch.tensor(gious)


        if self.loss_type == "iou":
            loss = 1 - ious
        elif self.loss_type == "giou":
            loss = 1 - gious
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
