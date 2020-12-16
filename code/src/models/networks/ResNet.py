"""
author: Antoine Spahr

date : 14.12.2020

----------

TO DO :
"""
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Define a Basic ResNet block with two convolutional layer.

        >-conv-bn-relu-conv-bn-+-relu->
         |_____________________|

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Build BasicBlock.
        ----------
        INPUT
            |---- in_channels (int) number of input channels to the block.
            |---- out_channels (int) number of output channels to the block.
            |---- stride (int) the stride to apply on the first convolution to downscale feature map.
        OUTPUT
            |---- BasicBlock (nn.Module)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the BasicBlock.
        ----------
        INPUT
            |---- x (torch.tensor) input with shape [B, In_channels, h, w]
        OUTPUT
            |---- out (torch.tensor) output with shape [B, Out_channels, h', w']
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    Define a Bottelneck ResNet block with three convolutional layer and expansion of channels.

        >-conv-bn-relu-conv-bn-relu-conv-bn-+-relu->
         |_____________conv-bn______________|

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Build Bottelneck block.
        ----------
        INPUT
            |---- in_channels (int) number of input channels to the block.
            |---- out_channels (int) number of output channels to the block.
            |---- stride (int) the stride to apply on the first convolution to downscale feature map.
        OUTPUT
            |---- Bottelneck (nn.Module)
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the Bottelneck block.
        ----------
        INPUT
            |---- x (torch.tensor) input with shape [B, In_channels, h, w]
        OUTPUT
            |---- out (torch.tensor) output with shape [B, Out_channels, h', w']
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    Define Resnets architecture by combining multiple Basic or Bottelneck blocks.
    """
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        """
        Build a ResNet.
        ----------
        INPUT
            |---- block (nn.Module) the block to use (Basic or Bottleneck).
            |---- num_blocks (list of int) list of 4 values describing the number of blocks to use in each layers.
            |---- num_classes (int) the number of output classes
            |---- input_channels (int) the number of channels in the input images.
        OUTPUT
            |---- ResNet (nn.Module)
        """
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Build a layer by combining multiple blocks.
        ----------
        INPUT
            |---- block (nn.Module) the block to use (Basic or Bottleneck).
            |---- out_channels (int) number of channels in output.
            |---- num_blocks (int) number of block to stack.
            |---- stride (int) the stride to use on the first block.
        OUTPUT
            |---- layer (nn.Module) a sequence of blocks.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet.
        ----------
        INPUT
            |---- x (torch.tensor) input with shape [B, Input_channels, H, W]
        OUTPUT
            |---- out (torch.tensor) output with shape [B, num_classes]
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)


def ResNet34(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels)


def ResNet50(num_classes=10, input_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels)


def ResNet101(num_classes=10, input_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, input_channels=input_channels)


def ResNet152(num_classes=10, input_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, input_channels=input_channels)

#%%
# import torchsummary
# net = ResNet101(num_classes=2, in_ch=1)
# torchsummary.summary(net, (1,256,256))
