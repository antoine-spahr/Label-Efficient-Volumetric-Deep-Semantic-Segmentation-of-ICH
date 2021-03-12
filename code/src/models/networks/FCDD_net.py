"""

"""
import torch.nn as nn
import torch.nn.functional as F
#from src.models.networks.FCDD_BaseNet import FCDDNet, BaseNet
from .FCDD_BaseNet import FCDDNet, BaseNet

class FCDD_CNN_VGG(FCDDNet):
    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'

        self.features = nn.Sequential(
            self._create_conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.conv_final = self._create_conv2d(512, 1, 1)

    def forward(self, x, ad=True):
        x = self.features(x)

        if ad:
            x = self.conv_final(x)

        return x

# import torchsummary
# import torch
#
# net = FCDD_CNN_VGG([1,256,256], bias=True)
# input = torch.rand(1,1,256,256).float().requires_grad_(False)
# out_feat = net(input)
# out_feat.shape
#
# net.receptive_upsample(out_feat, reception=True).shape

#torchsummary.summary(net, input_size=(1,256,256))
