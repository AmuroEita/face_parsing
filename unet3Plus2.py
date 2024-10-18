import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import unetConv2, DSConv, GhostConv, SELayer, MSCA, DSDilatedConv

#UNet 3+ 
class unet3Plus2(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, bilinear=True, feature_scale=4,
                 is_deconv=True, is_batchnorm=True):
        super(unet3Plus2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [32, 64, 128, 256, 512, 1024]
        # filters = [40, 80, 160, 320, 480, 640]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.se1 = SELayer(filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.se2 = SELayer(filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.se3 = SELayer(filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.se4 = SELayer(filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.se5 = SELayer(filters[4])
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = unetConv2(filters[4], filters[5], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 6
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        
        '''stage 5d'''
        # h1->512*512, hd4->32*32, Pooling 16 times
        self.h1_PT_hd5 = nn.MaxPool2d(16, 16, ceil_mode=True)
        self.h1_PT_hd5_conv = DSConv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd5_se = SELayer(self.CatChannels)
        self.h1_PT_hd5_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd5_relu = nn.ReLU(inplace=True)

        # h2->256*256, hd4->32*32, Pooling 8 times
        self.h2_PT_hd5 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h2_PT_hd5_conv = DSConv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd5_se = SELayer(self.CatChannels)
        self.h2_PT_hd5_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd5_relu = nn.ReLU(inplace=True)

        # h3->128*128, hd4->32*32, Pooling 4 times
        self.h3_PT_hd5 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h3_PT_hd5_conv = DSConv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd5_se = SELayer(self.CatChannels)
        self.h3_PT_hd5_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd5_relu = nn.ReLU(inplace=True)
        
        # h3->64*64, hd4->32*32, Pooling 2 times
        self.h4_PT_hd5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h4_PT_hd5_conv = DSConv(filters[3], self.CatChannels, 3, padding=1)
        self.h4_PT_hd5_se = SELayer(self.CatChannels)
        self.h4_PT_hd5_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_PT_hd5_relu = nn.ReLU(inplace=True)

        # h4->32*32, hd4->32*32, Concatenation
        self.h5_Cat_hd5_conv = DSConv(filters[4], self.CatChannels, 3, padding=1)
        self.h5_Cat_hd5_se = SELayer(self.CatChannels)
        self.h5_Cat_hd5_bn = nn.BatchNorm2d(self.CatChannels)
        self.h5_Cat_hd5_relu = nn.ReLU(inplace=True)

        # hd5->32*32, hd4->32*32, Upsample 2 times
        self.hd6_UT_hd5 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd6_UT_hd5_conv = DSConv(filters[5], self.CatChannels, 3, padding=1)
        self.hd6_UT_hd5_se = SELayer(self.CatChannels)
        self.hd6_UT_hd5_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd6_UT_hd5_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv5d_1 = DSConv(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.se5d_1 = SELayer(self.UpChannels)
        self.bn5d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu5d_1 = nn.ReLU(inplace=True)
        
        print("-----")
        # print(self.h1_PT_hd5.shape + "\n")
        # print(self.h2_PT_hd5.shape + "\n")
        # print(self.h3_PT_hd5.shape + "\n")
        # print(self.h4_PT_hd5.shape + "\n")
        # print(self.h5_Cat_hd5.shape + "\n")
        # print(self.hd6_UT_hd5.shape + "\n")

        
        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = DSConv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_se = SELayer(self.CatChannels)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = DSConv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_se = SELayer(self.CatChannels)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = DSConv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_se = SELayer(self.CatChannels)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = DSConv(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_se = SELayer(self.CatChannels)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)
        
        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_se = SELayer(self.CatChannels)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd6_UT_hd4 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd6_UT_hd4_conv = DSConv(filters[5], self.CatChannels, 3, padding=1)
        self.hd6_UT_hd4_se = SELayer(self.CatChannels)
        self.hd6_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd6_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = DSConv(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.se4d_1 = SELayer(self.UpChannels)
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)



        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = DSConv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_se = SELayer(self.CatChannels)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = DSConv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_se = SELayer(self.CatChannels)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = DSConv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_se = SELayer(self.CatChannels)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_se = SELayer(self.CatChannels)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_se = SELayer(self.CatChannels)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)
        
        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd6_UT_hd3 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd6_UT_hd3_conv = DSConv(filters[5], self.CatChannels, 3, padding=1)
        self.hd6_UT_hd3_se = SELayer(self.CatChannels)
        self.hd6_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd6_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = DSConv(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.se3d_1 = SELayer(self.UpChannels)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = DSConv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_se = SELayer(self.CatChannels)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = DSConv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_se = SELayer(self.CatChannels)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_se = SELayer(self.CatChannels)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_se = SELayer(self.CatChannels)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_se = SELayer(self.CatChannels)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)
        
        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd6_UT_hd2 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd6_UT_hd2_conv = DSConv(filters[5], self.CatChannels, 3, padding=1)
        self.hd6_UT_hd2_se = SELayer(self.CatChannels)
        self.hd6_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd6_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = DSConv(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.se2d_1 = SELayer(self.UpChannels)
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = DSConv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_se = SELayer(self.CatChannels)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_se = SELayer(self.CatChannels)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_se = SELayer(self.CatChannels)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_se = SELayer(self.CatChannels)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_se = SELayer(self.CatChannels)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd6_UT_hd1 = nn.Upsample(scale_factor=32, mode='bilinear')  # 14*14
        self.hd6_UT_hd1_conv = DSConv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd6_UT_hd1_se = SELayer(self.CatChannels)
        self.hd6_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd6_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = DSConv(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.se1d_1 = SELayer(self.UpChannels)
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = DSConv(self.UpChannels, n_classes, 3, padding=1)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total parameters counts u : {:,} \n".format(total_params))

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64
        h1 = self.se1(h1)
        
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128
        h2 = self.se2(h2)

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256
        h3 = self.se3(h3)

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512
        h4 = self.se4(h4)

        h5 = self.maxpool4(h4)
        h5 = self.conv5(h5)  # h4->40*40*512
        h5 = self.se5(h5)

        h6 = self.maxpool5(h5)
        hd6 = self.conv6(h6)  # h5->20*20*1024

        ## -------------Decoder-------------
        
        # 5
        h1_PT_hd5 = self.h1_PT_hd5_relu(self.h1_PT_hd5_bn(self.h1_PT_hd5_se(self.h1_PT_hd5_conv(self.h1_PT_hd5(h1)))))
        h2_PT_hd5 = self.h2_PT_hd5_relu(self.h2_PT_hd5_bn(self.h2_PT_hd5_se(self.h2_PT_hd5_conv(self.h2_PT_hd5(h2)))))
        h3_PT_hd5 = self.h3_PT_hd5_relu(self.h3_PT_hd5_bn(self.h3_PT_hd5_se(self.h3_PT_hd5_conv(self.h3_PT_hd5(h3)))))
        h4_PT_hd5 = self.h4_PT_hd5_relu(self.h4_PT_hd5_bn(self.h4_PT_hd5_se(self.h4_PT_hd5_conv(self.h4_PT_hd5(h4)))))
        h5_Cat_hd5 = self.h5_Cat_hd5_relu(self.h5_Cat_hd5_bn(self.h5_Cat_hd5_se(self.h5_Cat_hd5_conv(h5))))
        hd6_UT_hd5 = self.hd6_UT_hd5_relu(self.hd6_UT_hd5_bn(self.hd6_UT_hd5_se(self.hd6_UT_hd5_conv(self.hd6_UT_hd5(hd6)))))
        hd5 = self.relu5d_1(self.bn5d_1(self.se5d_1(self.conv5d_1(torch.cat((h1_PT_hd5, h2_PT_hd5, h3_PT_hd5, h4_PT_hd5, h5_Cat_hd5, hd6_UT_hd5), 1))))) # hd4->40*40*UpChannels  
        
        # 4
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_se(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_se(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_se(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_se(self.h4_Cat_hd4_conv(h4))))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_se(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))))
        hd6_UT_hd4 = self.hd6_UT_hd4_relu(self.hd6_UT_hd4_bn(self.hd6_UT_hd4_se(self.hd6_UT_hd4_conv(self.hd6_UT_hd4(hd6)))))
        hd4 = self.relu4d_1(self.bn4d_1(self.se4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4, hd6_UT_hd4), 1))))) # hd4->40*40*UpChannels

        # 3
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_se(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_se(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_se(self.h3_Cat_hd3_conv(h3))))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_se(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_se(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))))
        hd6_UT_hd3 = self.hd6_UT_hd3_relu(self.hd6_UT_hd3_bn(self.hd6_UT_hd3_se(self.hd6_UT_hd3_conv(self.hd6_UT_hd3(hd6)))))
        hd3 = self.relu3d_1(self.bn3d_1(self.se3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3, hd6_UT_hd3), 1))))) # hd3->80*80*UpChannels

        # 2
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_se(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_se(self.h2_Cat_hd2_conv(h2))))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_se(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_se(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd4_UT_hd2_se(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))))
        hd6_UT_hd2 = self.hd6_UT_hd2_relu(self.hd6_UT_hd2_bn(self.hd6_UT_hd2_se(self.hd6_UT_hd2_conv(self.hd6_UT_hd2(hd6)))))
        hd2 = self.relu2d_1(self.bn2d_1(self.se2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2, hd6_UT_hd2), 1))))) # hd2->160*160*UpChannels

        # 1
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_se(self.h1_Cat_hd1_conv(h1))))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_se(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_se(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_se(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_se(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))))
        
        ss = self.hd6_UT_hd1_conv(self.hd6_UT_hd1(hd6))
        
        hd6_UT_hd1 = self.hd6_UT_hd1_relu(self.hd6_UT_hd1_bn(self.hd6_UT_hd1_se(ss)))
        # hd6_UT_hd1 = self.hd6_UT_hd1_relu(self.hd6_UT_hd1_bn(self.hd6_UT_hd1_se(self.hd6_UT_hd1_conv(self.hd6_UT_hd1(hd6)))))
        hd1 = self.relu1d_1(self.bn1d_1(self.se1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1, hd6_UT_hd1), 1))))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return F.sigmoid(d1)

