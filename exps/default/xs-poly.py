#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn
import torch

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 8
        self.num_colors = 4
        self.num_apexes = 4
        self.depth = 0.67
        self.width = 0.75
        self.act = 'hswish'
        self.random_size = (10, 16)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

    def get_model(self, sublinear=False):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if getattr(self, "model", None) is None:
            
            in_channels = [256, 512, 1024]
            in_channels_head = [192, 384, 768]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_apexes,self.num_classes,self.num_colors, self.width, in_channels=in_channels_head, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
