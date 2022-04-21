from torch import nn
import torch

from .network_blocks import BaseConv, Focus, DWConv, BaseConv, ShuffleV2DownSampling, ShuffleV2Basic

class Shufflenet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("stage2", "stage3", "stage4"),
        in_channels=3,
        stem_out_channels=32,
        act="silu",
    ):
        super().__init__()
        stage_unit_repeat = [3, 7 ,3]
        # stage_unit_repeat = [2, 5 ,2]
        self.out_features = out_features
        # base_channels = int(wid_mul * 96)          # multiplier 0.25 for ShuffleNetV2 0.5X
        base_channels = int(wid_mul * 128)          # multiplier 0.25 for ShuffleNetV2 0.5X
        base_depth = max(round(dep_mul * 3), 1)    # 3

        self.stem = Focus(3, 24, ksize=3,stride=2, act=act,depthwise=True)
        # self.conv1 = DWConv(24, 32, ksize=1, stride=2,act=act)
        # self.conv1 = DWConv(24, 32, ksize=1, stride=1,act=act)
        # Conv(in_channels * 4, out_channels, ksize, stride, act=act)
        self.conv1 = DWConv(24, 32, ksize=3, stride=2,act=act,no_depth_act=True)

        self.stage2_list = [ShuffleV2DownSampling(32, base_channels * 2, act=act)]
        for _ in range(stage_unit_repeat[0]):
            self.stage2_list.append(ShuffleV2Basic(base_channels * 2, base_channels * 2, act=act))
        self.stage2 = nn.Sequential(*self.stage2_list)

        self.stage3_list = [ShuffleV2DownSampling(base_channels * 2,base_channels * 4, act=act)]
        for _ in range(stage_unit_repeat[1]):
            self.stage3_list.append(ShuffleV2Basic(base_channels * 4, base_channels * 4, act=act))
        self.stage3 = nn.Sequential(*self.stage3_list)

        self.stage4_list = [ShuffleV2DownSampling(base_channels * 4,base_channels * 8, act=act)]
        for _ in range(stage_unit_repeat[2]):
            self.stage2_list.append(ShuffleV2Basic(base_channels * 8, base_channels * 8, act=act))
        self.stage4 = nn.Sequential(*self.stage4_list)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.conv1(x)
        # print(x.shape)
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