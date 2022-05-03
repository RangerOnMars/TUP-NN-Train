#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .shufflenetv2 import Shufflenet
from .ghost_pan import GhostPAN
from .shufflev2_pafpn import ShufflePAFPN 



class ShuffleV2PAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("stage2", "stage3", "stage4"),
        in_channels=[256, 512, 1024],
        out_channels=512,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        # self.backbone = Shufflenet(channels=self.in_channels, act=act)
        self.backbone = Shufflenet(channels=self.in_channels, out_features=in_features,act=act)
        self.out_channels = in_channels[1]
        # self.out_channels = 
        Conv = DWConv if depthwise else BaseConv
        # self.neck = GhostPAN(self.in_channels, self.out_channels, depthwise, activation=act)
        # self.neck = GhostPAN([24, 64, 128, 256], self.out_channels, depthwise, activation=act)
        self.neck = ShufflePAFPN(self.in_channels, self.out_channels, depthwise, activation=act)

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
        # for i in range(len(features)):
        #     print(features[i].shape)
        # [x2, x1, x0] = features
        [x2, x1, x0] = features
        outputs = self.neck([x2, x1, x0])
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)
        # outputs = self.neck([x2,x1,x0])
        return outputs
    
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
        out_channels=512,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        # self.backbone = Shufflenet(channels=self.in_channels, act=act)
        self.backbone = Shufflenet(channels=self.in_channels, out_features=in_features,act=act)
        # self.out_channels = in_channels[1]
        self.out_channels = 96
        Conv = DWConv if depthwise else BaseConv
        # self.neck = GhostPAN(self.in_channels, self.out_channels, depthwise, activation=act)
        self.neck = GhostPAN([24, 64, 128, 256], self.out_channels, depthwise, activation=act)

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
        # for i in range(len(features)):
        #     print(features[i].shape)
        # [x2, x1, x0] = features
        [x2, x1, x0] = features
        outputs = self.neck([x2, x1, x0])
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)
        # outputs = self.neck([x2,x1,x0])
        return outputs