#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .shufflenetv2 import Shufflenet
from .ghost_pan import GhostPAN



class ShuffleV2GhostPAN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("stage2", "stage3", "stage4"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = Shufflenet(depth, width, act=act)
        self.in_features = in_features
        self.in_channels = [round(in_channels[0] * width), round(in_channels[1] * width), round(in_channels[2] * width)]
        self.out_channels = self.in_channels[0]
        Conv = DWConv if depthwise else BaseConv
        self.neck = GhostPAN(self.in_channels, self.out_channels, depthwise, activation=act)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        # print("x2",x2.shape)
        # print("x1",x1.shape)
        # print("x0",x0.shape)

        outputs = self.neck([x2,x1,x0])
        return outputs