from torch import nn
import torch

from .network_blocks import BaseConv, Focus, DWConv, BaseConv, ShuffleV2DownSampling, ShuffleV2Basic

class Shufflenet(nn.Module):
    def __init__(
        self,
        channels,
        out_features=("stage2", "stage3", "stage4"),
        act="silu",
    ):
        super().__init__()
        stage_unit_repeat = [3, 7 ,3]
        self.channels = []
        self.out_features = out_features
        base_channels = channels
        # print(chann)

        self.stem_list = []
        self.stem_list.append(DWConv(3,24,ksize=5,stride=2))
        # self.stem_list.append(DWConv(3, 48, ksize=7, stride=4,act=act))
        # self.stem_list.append(DWConv(3, 48, ksize=3, stride=2,act=act))
        # self.stem_list.append(DWConv(48, 48, ksize=3, stride=2,act=act))
        self.stem = nn.Sequential(*self.stem_list)
        
        self.conv1 = DWConv(24, 24, ksize=3, stride=2,act=act)

        self.stage2_list = [ShuffleV2DownSampling(24, base_channels[0], act=act)]
        for _ in range(stage_unit_repeat[0]):
            self.stage2_list.append(ShuffleV2Basic(base_channels[0], base_channels[0], act=act))
        self.stage2 = nn.Sequential(*self.stage2_list)

        self.stage3_list = [ShuffleV2DownSampling(base_channels[0], base_channels[1],act=act)]
        for _ in range(stage_unit_repeat[1]):
            self.stage3_list.append(ShuffleV2Basic(base_channels[1], base_channels[1], act=act))
        self.stage3 = nn.Sequential(*self.stage3_list)

        self.stage4_list = [ShuffleV2DownSampling(base_channels[1], base_channels[2], act=act)]
        for _ in range(stage_unit_repeat[2]):
            self.stage2_list.append(ShuffleV2Basic(base_channels[2], base_channels[2], act=act))
        self.stage4 = nn.Sequential(*self.stage4_list)

    def forward(self, x):
        outputs = {}
        # print(x.shape)
        x = self.stem(x)
        outputs["stem"] = x
        x = self.conv1(x)
        outputs["stage1"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        # print("stem", outputs["stem"].shape)
        # print("stage2", outputs["stage2"].shape)
        # print("stage3", outputs["stage3"].shape)
        # print("stage4", outputs["stage4"].shape)
        return {k: v for k, v in outputs.items() if k in self.out_features}
    