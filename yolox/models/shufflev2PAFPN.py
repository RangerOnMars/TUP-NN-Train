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
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):  
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.backbone = Shufflenet(channels=self.in_channels, out_features=in_features,act=act)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2]), int(in_channels[1]), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[1]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[0]), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0]),
            int(in_channels[0]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0]), int(in_channels[0]), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0]),
            int(in_channels[1]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1]), int(in_channels[1]), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[2]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

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

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

# class ShuffleV2PAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("stage2", "stage3", "stage4"),
#         in_channels=[256, 512, 1024],
#         out_channels=512,
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.in_channels = in_channels
#         # self.backbone = Shufflenet(channels=self.in_channels, act=act)
#         self.backbone = Shufflenet(channels=self.in_channels, out_features=in_features,act=act)
#         self.out_channels = in_channels[1]
#         # self.out_channels = 
#         Conv = DWConv if depthwise else BaseConv
#         self.neck = ShufflePAFPN(self.in_channels, self.out_channels, depthwise, activation=act)

#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """

#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         # for i in range(len(features)):
#         #     print(features[i].shape)
#         # [x2, x1, x0] = features
#         [x2, x1, x0] = features
#         outputs = self.neck([x2, x1, x0])
#         # print(outputs[0].shape)
#         # print(outputs[1].shape)
#         # print(outputs[2].shape)
#         # outputs = self.neck([x2,x1,x0])
#         return outputs
    
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

        self.backbone = Shufflenet(channels=self.in_channels, out_features=in_features,act=act)
        self.out_channels = 128
        Conv = DWConv if depthwise else BaseConv
        # self.neck = GhostPAN(self.in_channels, self.out_channels, depthwise, activation=act)
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