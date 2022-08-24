# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from this import s

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from ..libs.GatedSpatialConv import GatedSpatialConv2d as GCL
from ..libs.Resnet import BasicBlock as Block
from ..libs.RCAB import CALayer


import cv2
from ..config import vit_seg_configs as configs
from .ResNet50v2 import ResNetV2
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .vit import ViT
import torch.nn.functional as F
BatchNorm = SynchronizedBatchNorm2d
logger = logging.getLogger(__name__)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class GateBoundaryEnhancing(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = Block(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Block(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Block(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn7 = nn.Conv2d(1024, 1, 1)
        self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gate1 = GSC(32, 32)
        self.gate2 = GSC(16, 16)
        self.gate3 = GSC(8, 8)

    def forward(self, x, features):
        x_size = x.size()

        s3 = F.interpolate(self.dsn3(features[2]), x_size[2:],
                           mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(features[1]), x_size[2:],
                           mode='bilinear', align_corners=True)

        im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(features[3])
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)

        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        return edge_out, acts


class UP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            # nn.BatchNorm2d(128),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
        )
    def forward(self, x):
        x1 = self.up(x)

        return x1


class Down(nn.Module):
    def __init__(self, config):
        super(Down, self).__init__()
        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = self.hybrid_model(x)
        return features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=512, num_classes=1, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.Down = Down(config)
        self.transformer = ViT(config)
        self.decoder = UP()
        self.segmentation_head = SegmentationHead(
            in_channels= 64,
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.fuse = nn.Conv2d(1024,512,1,1)
        self.conv = nn.Conv2d(1, 64, 1)
        self.shape = GateBoundaryEnhancing()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.CAL = CALayer(1024)
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.Down(x)  # (B, n_patch, hidden)
        edge, acts = self.shape(x, features)
        edges = self.conv(acts)

        t0 = self.transformer(features[0])
        f0 = self.up(t0)
        f0 = self.fuse(f0)
        f1 = torch.cat((f0,features[1]),dim=1)
        f1 = self.CAL(f1)
        seg = self.decoder(f1)
        seg = seg + edges
        
        logits = self.segmentation_head(seg)
        return logits,acts



CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),  #
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}