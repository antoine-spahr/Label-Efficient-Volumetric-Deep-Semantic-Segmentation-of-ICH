"""
author: Antoine Spahr

date : 25.09.2020

----------

TO DO :
- Check if double conv filters should be x --> 2x --> 2x (as in original Unet) or x --> x --> 2x (as in Unet 3D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    U-Net model as a Pytorch nn.Module. The class enables to build 2D and 3D U-Nets with different depth.
    """
    def __init__(self, depth=5, use_3D=False, bilinear=False, in_channels=1, out_channels=1, top_filter=64):
        """
        Build a 2D or 3D U-Net Module.
        ----------
        INPUT
            |---- depth (int) the number of double convolution block to use including the bottleneck double-convolution.
            |---- use_3D (bool) whether the U-Net should be built with 3D Convolution, Pooling, etc layers.
            |---- bilinear (bool) whether to use Upsampling layer in the synthesis path, otherwise transposed convolutions
            |           are used. If True, the upward double convolutional block takes care of reducing the number of
            |           channels instead of the transposed convolution.
            |---- in_channels (int) the number of input channels to the U-Net
            |---- out_channels (int) the number of output channels (i.e. the number of segemtnation classes).
            |---- top_filter (int) the number of channels after the first double-convolution block.
        OUTPUT
            |---- U-Net (nn.Module) The U-Net.
        """
        super(UNet, self).__init__()
        # Initialize Modules Lists for the encoder, decoder, and upsampling blocks
        self.down_block = nn.ModuleList()
        self.up_samp = nn.ModuleList()
        self.up_block = nn.ModuleList()
        # define filter number for each block, after each down block the number of filters double, after each up block the number of filter is reduced
        down_filter_list = [(in_channels, top_filter)] + [(top_filter*(2**d), top_filter*(2**(d+1))) for d in range(depth-2)]
        bottleneck_filter = (top_filter*(2**(depth-2)), top_filter*(2**(depth-1)))
        up_filter_list = [(top_filter*(2**d), top_filter*(2**(d-1))) for d in range(depth-1, 0, -1)]

        # initialize blocks
        for down_ch, up_ch in zip(down_filter_list, up_filter_list):
            self.down_block.append(ConvBlock(down_ch[0], down_ch[1], mid_channels=down_ch[1]//2, use_3D=use_3D))
            # Define the synthesis strategy (trasnposed convolution vs upsampling)
            if bilinear:
                self.up_block.append(ConvBlock(int(1.5*up_ch[0]), up_ch[1], mid_channels=up_ch[1], use_3D=use_3D))
                up_mode = 'trilinear' if use_3D else 'bilinear'
                self.up_samp.append(nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True))
            else:
                self.up_block.append(ConvBlock(up_ch[0], up_ch[1], mid_channels=up_ch[1], use_3D=use_3D))
                convT = nn.ConvTranspose3d if use_3D else nn.ConvTranspose2d
                self.up_samp.append(convT(up_ch[0], up_ch[1], kernel_size=2, stride=2))

        # bottelneck convolutional block
        self.bottleneck_block = ConvBlock(bottleneck_filter[0], bottleneck_filter[1], mid_channels=bottleneck_filter[1]//2, use_3D=use_3D)
        # Down pooling module
        self.downpool = nn.MaxPool3d(kernel_size=2, stride=2) if use_3D else nn.MaxPool2d(kernel_size=2, stride=2)
        # define the final convolution (1x1(x1)) convolution follow by a sigmoid.
        self.final_conv = nn.Conv3d(top_filter, out_channels, kernel_size=1) if use_3D else nn.Conv2d(top_filter, out_channels, kernel_size=1)
        if out_channels > 1:
            self.final_activation = nn.Softmax(dim=1) # Softmax along output channels for multiclass predictions
        else:
            self.final_activation = nn.Sigmoid() # Sigmoid for logit

    def forward(self, input):
        """
        Forward pass of the U-Net. Encoding and decoding with skip connections.
        ----------
        INPUT
            |---- input (torch.Tensor) input to U-Net with dimension (Batch x Channels x Height x Width (x Depth)). The
            |           input is 4D for U-Net2D and 5D for U-Net3D.
        OUTPUT
            |---- out (torch.Tensor) the segmentation output with dimension (Batch x N_classes x Height x Width (x Depth)).
        """
        # Encoder
        x = input
        res_list = []
        for conv in self.down_block:
            x = conv(x)
            res_list.append(x)
            x = self.downpool(x)

        # bottleneck convolution
        x = self.bottleneck_block(x)

        # Decoder
        for up, conv, res in zip(self.up_samp, self.up_block, res_list[::-1]):
            x = up(x) # convTranspose
            x = conv(torch.cat([res, x], dim=1)) # concat + convolutional block

        # final 1x1 convolution
        out = self.final_activation(self.final_conv(x))

        return out

class ConvBlock(nn.Module):
    """
    Double convolution modules used for each block of the U-Nets. It's composed of a succesion of two [Conv -> BN -> ReLU].
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, use_3D=False):
        """
        Build a Double Convolution block.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the block.
            |---- out_channels (int) the number of output channels of the block.
            |---- mid_channels (int) the number of channel after the first convolution. If not provided, it's set to out_channels.
            |---- kernel_size (int) the kernel size to use in the convolution layers.
            |---- use_3D (bool) whether the layers use must be for volumetric inputs.
        OUTPUT
            |---- ConvBlock (nn.Modules) a double convoltution module.
        """
        super(ConvBlock, self).__init__()
        self.activation = nn.ReLU()
        mid_channels = mid_channels if mid_channels else out_channels
        if use_3D:
            self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1)
            self.bn1 = nn.BatchNorm3d(mid_channels)
            self.conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
            self.bn2 = nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        """
        Forward pass of the double convolution block.
        ----------
        INPUT
            |---- input (torch.Tensor) the input to the block with dimension (Batch x In_Channels x Height x Width (x Depth)).
            |           The input is 4D for the U-Net2D or 5D for the U-Net3D.
        OUTPUT
            |---- out (torch.Tensor) the output of the block with dimension (Batch x Out_Channels x Height x Width (x Depth)).
        """
        x = self.activation(self.bn1(self.conv1(input)))
        out = self.activation(self.bn2(self.conv2(x)))

        return out
