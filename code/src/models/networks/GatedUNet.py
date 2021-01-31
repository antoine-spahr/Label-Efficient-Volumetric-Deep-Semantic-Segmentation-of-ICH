import torch
import torch.nn as nn
import torch.nn.functional as F

#from src.models.networks.InpaintingNetwork import GatedConv2d, UpsampleGatedConv2d

class UNet(nn.Module):
    """
    U-Net model as a Pytorch nn.Module. The class enables to build 2D and 3D U-Nets with different depth.
    """
    def __init__(self, depth=5, use_3D=False, bilinear=False, in_channels=1, out_channels=1, top_filter=64, midchannels_factor=2,
                 p_dropout=0.5, use_final_activation=True, use_gatedConv=False):
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
            |---- midchannels_factor (int) defines the number of mid-channels in the convolutional blocks as
            |           mid_channels = out_channels // midchannels_factor.
            |---- p_dropout (float or list) dropout probability in down block. If a float is passed, the same dropout
            |           probability is applied to each down convolutional block. If a list is passed, it should have a
            |           length equal depth which specify the dropout probability for each block.
            |---- use_final_activation (bool) whether to use a final activation layer (Simoid or Softmax).
        OUTPUT
            |---- U-Net (nn.Module) The U-Net.
        """
        super(UNet, self).__init__()
        # check if dropout input
        if isinstance(p_dropout, float):
            p_dropout_list = [p_dropout]*depth
        elif isinstance(p_dropout, list):
            assert len(p_dropout) == depth, f'p_dropout provided as list should have the same length as depth. p_dropout {len(p_dropout)} vs depth {depth}.'
            p_dropout_list = p_dropout
        else:
            raise TypeError(f'p_dropout list not supported. Should be float or list of float. Given {type(p_dropout)}.')
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
        for down_ch, up_ch, p_drop in zip(down_filter_list, up_filter_list, p_dropout_list[:-1]):
            self.down_block.append(ConvBlock(down_ch[0], down_ch[1], mid_channels=down_ch[1]//midchannels_factor, use_3D=use_3D, p_dropout=p_drop, use_gatedConv=use_gatedConv))
            # Define the synthesis strategy (trasnposed convolution vs upsampling)
            if bilinear or use_gatedConv:
                self.up_block.append(ConvBlock(int(1.5*up_ch[0]), up_ch[1], mid_channels=up_ch[1], use_3D=use_3D, use_gatedConv=use_gatedConv))
                up_mode = 'trilinear' if use_3D else 'bilinear'
                self.up_samp.append(nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True))
            else:
                self.up_block.append(ConvBlock(up_ch[0], up_ch[1], mid_channels=up_ch[1], use_3D=use_3D, use_gatedConv=False))
                convT = nn.ConvTranspose3d if use_3D else nn.ConvTranspose2d
                self.up_samp.append(convT(up_ch[0], up_ch[1], kernel_size=2, stride=2))

        # bottelneck convolutional block
        self.bottleneck_block = ConvBlock(bottleneck_filter[0], bottleneck_filter[1], mid_channels=bottleneck_filter[1]//midchannels_factor,
                                          use_3D=use_3D, p_dropout=p_dropout_list[-1], use_gatedConv=use_gatedConv)
        # Down pooling module
        self.downpool = nn.MaxPool3d(kernel_size=2, stride=2) if use_3D else nn.MaxPool2d(kernel_size=2, stride=2)
        # define the final convolution (1x1(x1)) convolution follow by a sigmoid.
        if use_gatedConv:
            self.final_conv = GatedConv(top_filter, out_channels, kernel_size=1, use_3D=use_3D, activation='none', batch_norm=False)
        else:
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
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, use_3D=False, p_dropout=0.0, use_gatedConv=False):
        """
        Build a Double Convolution block.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the block.
            |---- out_channels (int) the number of output channels of the block.
            |---- mid_channels (int) the number of channel after the first convolution. If not provided, it's set to out_channels.
            |---- kernel_size (int) the kernel size to use in the convolution layers.
            |---- use_3D (bool) whether the layers use must be for volumetric inputs.
            |---- p_dropout (float) probability of dropout. If 0.0 no dropout applied.
        OUTPUT
            |---- ConvBlock (nn.Modules) a double convoltution module.
        """
        super(ConvBlock, self).__init__()
        assert 0.0 <= p_dropout <= 1.0, f'Dropout probaility must be in [0.0, 1.0]. Given {p_dropout}.'
        self.dropout = nn.Dropout(p=p_dropout)

        mid_channels = mid_channels if mid_channels else out_channels
        #conv_fn = GatedConv if use_gatedConv else ConvLayer

        if use_gatedConv:
            self.conv1 = GatedConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1,
                                   bias=True, padding_mode='zeros', activation='relu', batch_norm=True, use_3D=use_3D)
            self.conv2 = GatedConv(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                                   bias=True, padding_mode='zeros', activation='relu', batch_norm=True, use_3D=use_3D)
        else:
            self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1,
                                   bias=True, padding_mode='zeros', activation='relu', batch_norm=True, use_3D=use_3D)
            self.conv2 = ConvLayer(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                                   bias=True, padding_mode='zeros', activation='relu', batch_norm=True, use_3D=use_3D )

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
        x = self.conv1(input)
        out = self.conv2(x)
        if self.dropout.p > 0.0:
            out = self.dropout(out)
        return out

class ConvLayer(nn.Module):
    """
    Define a 2D Convolution Layer with possibility to add spectral Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 padding_mode='zeros', activation='relu', batch_norm=True, use_3D=False, sn=False, power_iter=1):
        """
        Build a 2D (or 3D) convolutional layer as a pytorch nn.Module.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the convolutions.
            |---- out_channels (int) the number of output channels of the convolutions.
            |---- kernel_size (int) the kernel size of the convolutions.
            |---- stride (int) the stride of the convolution kernel.
            |---- padding (int) the padding of the input prior to the convolution.
            |---- dilation (int) the dilation of the kernel.
            |---- bias (bool) wether to use a bias term on the convolution kernel.
            |---- padding_mode (str) how to pad the image (see nn.Conv2d doc for details).
            |---- activation (str) the activation function to use. Supported: 'relu' -> ReLU, 'lrelu' -> LeakyReLU,
            |               'prelu' -> PReLU, 'selu' -> SELU, 'tanh' -> Hyperbolic tangent, 'sigmoid' -> sigmoid,
            |               'none' -> No activation used
            |---- batch_norm (bool) whether to use a batch normalization layer between the convolution and the activation.
            |---- sn (bool) whether to use Spetral Normalization on the convolutional weights.
            |---- power_iter (int) the number of iteration for Spectral norm estimation.
        OUTPUT
            |---- ConvLayer (nn.Module) the convolution layer.
        """
        super(ConvLayer, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"

        conv_fn = nn.Conv3d if use_3D else nn.Conv2d
        batchnorm_fn = nn.BatchNorm3d if use_3D else nn.BatchNorm2d

        if sn:
            self.conv = nn.utils.spectral_norm(conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                                         dilation=dilation, bias=bias, padding_mode=padding_mode),
                                               n_power_iterations=power_iter)
        else:
            self.conv = conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  bias=bias, padding_mode=padding_mode)

        self.norm = batchnorm_fn(out_channels) if batch_norm else None

    def forward(self, x):
        """
        Forward pass of the GatedConvolution Layer.
        ----------
        INPUT
            |---- x (torch.tensor) input with dimension (Batch x in_Channel x H x W).
        OUTPUT
            |---- out (torch.tensor) the output with dimension (Batch x out_Channel x H' x W').
        """
        # Conv->(BN)->Activation
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class GatedConv(nn.Module):
    """
    Define a Gated 2D (or 3D) Convolution as proposed in Yu et al. 2018. (Free-Form Image Inpainting with Gated Convolution).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 padding_mode='zeros', activation='relu', batch_norm=True, use_3D=False):
        """
        Build a Gated 2D convolutional layer as a pytorch nn.Module.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the convolutions.
            |---- out_channels (int) the number of output channels of the convolutions.
            |---- kernel_size (int) the kernel size of the convolutions.
            |---- stride (int) the stride of the convolution kernel.
            |---- padding (int) the padding of the input prior to the convolution.
            |---- dilation (int) the dilation of the kernel.
            |---- bias (bool) wether to use a bias term on the convolution kernel.
            |---- padding_mode (str) how to pad the image (see nn.Conv2d doc for details).
            |---- activation (str) the activation function to use. Supported: 'relu' -> ReLU, 'lrelu' -> LeakyReLU,
            |               'prelu' -> PReLU, 'selu' -> SELU, 'tanh' -> Hyperbolic tangent, 'sigmoid' -> sigmoid,
            |               'none' -> No activation used
            |---- batch_norm (bool) whether to use a batch normalization layer between the convolution and the activation.
        OUTPUT
            |---- GatedConv (nn.Module) the gated convolution layer.
        """
        super(GatedConv, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"

        conv_fn = nn.Conv3d if use_3D else nn.Conv2d
        batchnorm_fn = nn.BatchNorm3d if use_3D else nn.BatchNorm2d

        self.conv_feat = conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                 bias=bias, padding_mode=padding_mode)
        self.conv_gate = conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                 bias=bias, padding_mode=padding_mode)
        self.sigmoid = nn.Sigmoid()
        self.norm = batchnorm_fn(out_channels) if batch_norm else None

    def forward(self, x):
        """
        Forward pass of the GatedConvolution Layer.
        ----------
        INPUT
            |---- x (torch.tensor) input with dimension (Batch x in_Channel x H x W).
        OUTPUT
            |---- out (torch.tensor) the output with dimension (Batch x out_Channel x H' x W').
        """
        # Conv->(BN)->Activation
        feat = self.conv_feat(x)
        if self.norm:
            feat = self.norm(feat)
        if self.activation:
            feat = self.activation(feat)
        # gating
        gate = self.sigmoid(self.conv_gate(x))
        # output
        out = feat * gate
        return out

# %%
# import torchsummary
#
# unet = UNet(depth=5, use_3D=False, in_channels=2, out_channels=1, top_filter=32, midchannels_factor=2, p_dropout=0.0,
#             use_final_activation=True, use_gatedConv=True, bilinear=False)
#
# torchsummary.summary(unet, (2,256,256))
