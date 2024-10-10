import torch.nn as nn

from model_utils import *

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class unet_plus(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.cbam1 = CBAM(filters[0])
        self.maxpool1 = nn.Conv(filters[0], filters[0], 1, 2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.cbam2 = CBAM(filters[0])
        self.maxpool2 = nn.Conv(filters[0], filters[0], 1, 2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.cbam3 = CBAM(filters[0])
        self.maxpool3 = nn.Conv(filters[0], filters[0], 1, 2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.cbam4 = CBAM(filters[0])
        self.maxpool4 = nn.Conv(filters[0], filters[0], 1, 2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total parameters count: {:,} \n".format(total_params))

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        cbam1 = self.cbam1
        conv1 = conv1 + cbam1
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        cbam2 = self.cbam2
        conv2 = conv2 + cbam2
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        cbam3 = self.cbam3
        conv3 = conv3 + cbam3
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        cbam4 = self.cbam4
        conv4 = conv4 + cbam4
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    def get_params_count(self):
        
        return sum(p.numel() for p in model.parameters() if p.requires_grad)