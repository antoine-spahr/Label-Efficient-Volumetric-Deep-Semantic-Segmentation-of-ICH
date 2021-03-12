"""
author: Antoine Spahr

date : 01.03.2021

----------
To Do:
    -
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """ Encoder """
    def __init__(self, in_channels=1, latent_channels=64, bottelneck_channels=64, n_conv=3, kernel_size=5):
        """

        """
        super(Encoder, self).__init__()
        in_list = [latent_channels*2**i for i in range(n_conv)]
        out_list = [c*2 for c in in_list]
        self.in_conv = nn.Sequential(nn.Conv2d(in_channels, latent_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                     nn.BatchNorm2d(latent_channels), nn.ReLU())
        self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU()) for in_ch, out_ch in zip(in_list, out_list)])
        self.bottelneck_conv = nn.Sequential(nn.Conv2d(out_list[-1], bottelneck_channels, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(bottelneck_channels), nn.ReLU())

    def forward(self, x):
        """

        """
        x = self.in_conv(x)
        for conv in self.conv_list:
            x = conv(x)
        x = self.bottelneck_conv(x)

        return x

class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, bottelneck_channels=64, latent_channels=64, out_channels=1, n_conv=3, bilinear=False, kernel_size=5):
        """

        """
        super(Decoder, self).__init__()
        in_list = [latent_channels*2**(i+1) for i in range(n_conv)][::-1]
        out_list = [c//2 for c in in_list]
        if bilinear:
            self.bottelneck_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                 nn.Conv2d(bottelneck_channels, in_list[0], kernel_size=3, stride=1, padding=1),
                                                 nn.BatchNorm2d(in_list[0]), nn.ReLU())
            self.conv_list = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                          nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                                          nn.BatchNorm2d(out_ch),
                                                          nn.ReLU()) for in_ch, out_ch in zip(in_list, out_list)])
        else:
            self.bottelneck_conv = nn.Sequential(nn.ConvTranspose2d(bottelneck_channels, in_list[0], kernel_size=2, stride=2, padding=0),
                                                 nn.BatchNorm2d(in_list[0]), nn.ReLU())
            self.conv_list = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size-1, stride=2, padding=(kernel_size-2)//2),
                                            nn.BatchNorm2d(out_ch),
                                            nn.ReLU()) for in_ch, out_ch in zip(in_list, out_list)])
        self.out_conv = nn.Sequential(nn.Conv2d(latent_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                     nn.BatchNorm2d(out_channels), nn.Tanh())

    def forward(self, x):
        """
        """
        x = self.bottelneck_conv(x)
        for convT in self.conv_list:
            x = convT(x)
        x = self.out_conv(x)

        return x

class AE_net(nn.Module):
    """ AE """
    def __init__(self, in_channels=1, latent_channels=64, bottelneck_channels=64, n_conv=3, bilinear=False, kernel_size=5):
        """
        Auto-Encoder network.
        ----------
        INPUT
            |---- in_channels (int) number of input/output channels.
            |---- latent_channels (int) number of channels after the first convolutions. Number of channels downstream
            |               will be half of the top one.
            |---- bottelneck_channels (int) number of channels in the bottleneck.
            |---- n_conv (int) number of convolution block [Conv-BN-ReLU] between the input and the bottelneck.
            |---- bilinear (bool) whether to use Bilinear upsampling + Conv in decoder or transposed convolutions.
        OUTPUT
            |---- AE_net (nn.Module) the AE module.
        """
        super(AE_net, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_channels=latent_channels,
                               bottelneck_channels=bottelneck_channels, n_conv=n_conv, kernel_size=kernel_size)
        self.decoder = Decoder(bottelneck_channels=bottelneck_channels, latent_channels=latent_channels,
                               out_channels=in_channels, n_conv=n_conv, bilinear=bilinear, kernel_size=kernel_size)

    def forward(self, x):
        """

        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x

#%%
# import torchsummary
# net = AE_net(in_channels=1, latent_channels=64, bottelneck_channels=64, n_conv=3, bilinear=True, kernel_size=3)
# torchsummary.summary(net, input_size=(1,256,256), batch_size=-1)
