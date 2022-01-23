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
        # stage_unit_repeat = [3, 7 ,3]
        stage_unit_repeat = [2, 5 ,2]
        self.out_features = out_features
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.conv1 = DWConv(base_channels, base_channels * 2, ksize=3, stride=2,act=act)

        self.stage2_list = [ShuffleV2DownSampling(base_channels * 2,base_channels * 4, act=act)]
        for _ in range(stage_unit_repeat[0]):
            self.stage2_list.append(ShuffleV2Basic(base_channels * 4, base_channels * 4, act=act))
        self.stage2 = nn.Sequential(*self.stage2_list)

        self.stage3_list = [ShuffleV2DownSampling(base_channels * 4,base_channels * 8, act=act)]
        for _ in range(stage_unit_repeat[1]):
            self.stage3_list.append(ShuffleV2Basic(base_channels * 8, base_channels * 8, act=act))
        self.stage3 = nn.Sequential(*self.stage3_list)

        self.stage4_list = [ShuffleV2DownSampling(base_channels * 8,base_channels * 16, act=act)]
        for _ in range(stage_unit_repeat[2]):
            self.stage2_list.append(ShuffleV2Basic(base_channels * 16, base_channels * 16, act=act))
        self.stage4 = nn.Sequential(*self.stage4_list)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.conv1(x)
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}