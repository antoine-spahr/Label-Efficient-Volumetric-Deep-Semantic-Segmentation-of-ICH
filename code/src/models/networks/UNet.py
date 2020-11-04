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
    def __init__(self, depth=5, use_3D=False, bilinear=False, in_channels=1, out_channels=1, top_filter=64, use_final_activation=True):
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
            |---- use_final_activation (bool) whether to use a final activation layer (Simoid or Softmax).
        OUTPUT
            |---- U-Net (nn.Module) The U-Net.
        """
        super(UNet, self).__init__()
        # Network attribute to return or not the bottleneck representation
        self.return_bottleneck = False
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
        if use_final_activation:
            if out_channels > 1:
                self.final_activation = nn.Softmax(dim=1) # Softmax along output channels for multiclass predictions
            else:
                self.final_activation = nn.Sigmoid() # Sigmoid for logit
        else:
            self.final_activation = nn.Identity()

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
        if self.return_bottleneck:
            x_bottleneck = x

        # Decoder
        for up, conv, res in zip(self.up_samp, self.up_block, res_list[::-1]):
            x = up(x) # convTranspose
            x = conv(torch.cat([res, x], dim=1)) # concat + convolutional block

        # final 1x1 convolution
        out = self.final_activation(self.final_conv(x))

        if self.return_bottleneck:
            return out, x_bottleneck
        else:
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

class MLPHead(nn.Module):
    """
    Define a MLP projection head. Combination of Linear and ReLU
    """
    def __init__(self, Neurons_layer=[512,256,128]):
        """
        Build a MLP Head with the given structure.
        ----------
        INPUT
            |---- Neurons_layer (list of int) the Projection head structure. Each entry defines the number of neurons of
            |           a given layer of the MLP.
        OUTPUT
            |---- MLP_head (nn.Module) the MLP head
        """
        nn.Module.__init__(self)
        self.fc_layers = nn.ModuleList(nn.Linear(in_features=n_in, out_features=n_out) for n_in, n_out in zip(Neurons_layer[:-1], Neurons_layer[1:]))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the MLP projection head.
        ----------
        INPUT
            |---- input (torch.Tensor) input to MLP head with dimension (Batch x input_size).
        OUTPUT
            |---- out (torch.Tensor) the output with dimension (Batch x output_size).
        """
        for linear in self.fc_layers[:-1]:
            x = self.relu(linear(x))
        x = self.fc_layers[-1](x)
        return x

class ConvHead(nn.Module):
    """
    Define a MLP projection head. Combination of Linear and ReLU
    """
    def __init__(self, channel_layer=[128, 256, 32], use_3D=False):
        """
        Build a Convolutional Head with the given structure of conv1x1 channels.
        ----------
        INPUT
            |---- channel_layer (list of int) the convolutional head structure. Each entry defines the number of channels of
            |           a given 1x1 convolution layer.
            |---- use_3D (bool) whether to use 3D convolution or 2D.
        OUTPUT
            |---- Conv_head (nn.Module) the convolutional head
        """
        super(ConvHead, self).__init__()
        conv = nn.Conv3d if use_3D else nn.Conv2d
        self.conv_layers = nn.ModuleList(conv(in_channels=n_in, out_channels=n_out, kernel_size=1) for n_in, n_out in zip(channel_layer[:-1], channel_layer[1:]))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the convolutional projection head.
        ----------
        INPUT
            |---- input (torch.Tensor) input to convolutional head with dimension (Batch x channel_layer[0] x input dim).
        OUTPUT
            |---- out (torch.Tensor) the output with dimension (Batch x channel_layer[-1] x output dim).
        """
        for linear in self.conv_layers[:-1]:
            x = self.relu(linear(x))
        x = self.conv_layers[-1](x)
        return x

class UNet_Encoder(nn.Module):
    """
    U-Net model as a Pytorch nn.Module. The class enables to build 2D and 3D U-Nets with different depth.
    """
    def __init__(self, depth=5, use_3D=False, in_channels=1, MLP_head=[256, 128], top_filter=64):
        """
        Build a 2D or 3D U-Net Module.
        ----------
        INPUT
            |---- depth (int) the number of double convolution block to use including the bottleneck double-convolution.
            |---- use_3D (bool) whether the U-Net should be built with 3D Convolution, Pooling, etc layers.
            |---- in_channels (int) the number of input channels to the U-Net
            |---- MLP_head (list of int) the structure of the additional MLP head to the convolutional encoder.
            |---- top_filter (int) the number of channels after the first double-convolution block.
        OUTPUT
            |---- U-Net Encoder (nn.Module) The U-Net Encoder.
        """
        super(UNet_Encoder, self).__init__()
        # Network attribute to return or not the bottleneck representation
        self.return_bottleneck = False
        # Initialize Modules Lists for the encoder, decoder, and upsampling blocks
        self.down_block = nn.ModuleList()
        # define filter number for each block, after each down block the number of filters double, after each up block the number of filter is reduced
        down_filter_list = [(in_channels, top_filter)] + [(top_filter*(2**d), top_filter*(2**(d+1))) for d in range(depth-2)]
        bottleneck_filter = (top_filter*(2**(depth-2)), top_filter*(2**(depth-1)))

        # initialize blocks
        for down_ch in down_filter_list:#, up_ch in zip(down_filter_list, up_filter_list):
            self.down_block.append(ConvBlock(down_ch[0], down_ch[1], mid_channels=down_ch[1]//2, use_3D=use_3D))

        # bottelneck convolutional block
        self.bottleneck_block = ConvBlock(bottleneck_filter[0], bottleneck_filter[1], mid_channels=bottleneck_filter[1]//2, use_3D=use_3D)
        # Down pooling module
        self.downpool = nn.MaxPool3d(kernel_size=2, stride=2) if use_3D else nn.MaxPool2d(kernel_size=2, stride=2)
        # MLP Head
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1)) if use_3D else nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_head = MLPHead(Neurons_layer=[bottleneck_filter[1]] + MLP_head)

    def forward(self, input):
        """
        Forward pass of the U-Net-like Encoder.
        ----------
        INPUT
            |---- input (torch.Tensor) input to U-Net with dimension (Batch x Channels x Height x Width (x Depth)). The
            |           input is 4D for U-Net2D and 5D for U-Net3D.
        OUTPUT
            |---- out (torch.Tensor) input representation with dimension (Batch x output_size).
        """
        # Encoder
        x = input
        #res_list = []
        for conv in self.down_block:
            x = conv(x)
            #res_list.append(x)
            x = self.downpool(x)
        # bottleneck convolution
        x = self.bottleneck_block(x)
        # MLP projection head
        x = self.avg_pool(x)
        if self.return_bottleneck:
            x_bottleneck = x
        out = self.mlp_head(torch.flatten(x, 1))

        if self.return_bottleneck:
            return out, x_bottleneck
        else:
            return out

class Partial_UNet(nn.Module):
    """
    A partial U-Net model as a Pytorch nn.Module. It's composed of a U-Net encoder and only a partial decoder (to be used
    in the local contrastive pretraining of Chaitanya 2020).
    """
    def __init__(self, depth=5, n_decoder=3, use_3D=False, bilinear=False, in_channels=1, head_channel=[64, 32], top_filter=64):
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
            |---- head_channel (list of int) the channel to use for the 1x1 convolution after the partial decoding. (length
            |           of list defines the number of 1x1 convolutions)
            |---- top_filter (int) the number of channels after the first double-convolution block.
        OUTPUT
            |---- U-Net (nn.Module) The U-Net.
        """
        super(Partial_UNet, self).__init__()
        # Network attribute to return or not the bottleneck representation
        self.return_bottleneck = False
        # Initialize Modules Lists for the encoder, decoder, and upsampling blocks
        self.down_block = nn.ModuleList()
        self.up_samp = nn.ModuleList()
        self.up_block = nn.ModuleList()
        # define filter number for each block, after each down block the number of filters double, after each up block the number of filter is reduced
        down_filter_list = [(in_channels, top_filter)] + [(top_filter*(2**d), top_filter*(2**(d+1))) for d in range(depth-2)]
        bottleneck_filter = (top_filter*(2**(depth-2)), top_filter*(2**(depth-1)))
        # only deconvolves partially
        up_filter_list = [(top_filter*(2**d), top_filter*(2**(d-1))) for d in range(depth-1, depth-1-n_decoder, -1)]
        self.n_decoder = n_decoder

        # initialize down blocks
        for down_ch in down_filter_list: #down_ch, up_ch in zip(down_filter_list, up_filter_list):
            self.down_block.append(ConvBlock(down_ch[0], down_ch[1], mid_channels=down_ch[1]//2, use_3D=use_3D))
        # initialize up block
        for up_ch in up_filter_list:
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
        # define the final layer as a combbination of 1x1 convolution (~local MLP)
        self.final_conv = ConvHead(channel_layer=[up_filter_list[-1][1]] + head_channel, use_3D=use_3D)

    def forward(self, input):
        """
        Forward pass of the partial U-Net. Encoding and decoding with skip connections.
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
        if self.return_bottleneck:
            x_bottleneck = x

        # Partial Decoder
        for up, conv, res in zip(self.up_samp, self.up_block, res_list[::-1][:self.n_decoder]):
            x = up(x) # convTranspose
            x = conv(torch.cat([res, x], dim=1)) # concat + convolutional block

        # final 1x1 convolution head
        out = self.final_conv(x)

        if self.return_bottleneck:
            return out, x_bottleneck
        else:
            return out
