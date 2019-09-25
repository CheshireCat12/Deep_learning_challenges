#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:54:28 2019

@author: cheshirecat12
"""
import torch
import torch.nn as nn

import torchvision.models as models


class SegnetDec(nn.Module):

    def __init__(self, in_channels, out_channels):

        self.layers = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels, in_channels//2, (3, 3), padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(),
                nn.Conv2d(in_channels//2, in_channels//2, (3, 3), padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(),
                nn.Conv2d(in_channels//2, out_channels, (3, 3), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Segnet(nn.Module):

    def __init__(self, num_classes):
        self.num_classes = num_classes

        encoder = list(models.vgg16_bn(pretrained=True).feature.children())

        self.enc1 = nn.Sequential(*encoder[:7])
        self.enc3 = nn.Sequential(*encoder[14:24])
        self.enc2 = nn.Sequential(*encoder[7:14])
        self.enc4 = nn.Sequential(*encoder[24:34])
        self.enc5 = nn.Sequential(*encoder[34:])

        self.dec5 = SegnetDec(512, 512)
        self.dec4 = SegnetDec(1024, 256)
        self.dec3 = SegnetDec(512, 128)
        self.dec2 = SegnetDec(256, 64)
        self.dec1 = SegnetDec(128, 32)

        self.final = nn.Conv2d(32, self.num_classes, (3, 3), padding=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.concat(enc4, dec5, 1))
        dec3 = self.dec3(torch.concat(enc3, dec4, 1))
        dec2 = self.dec2(torch.concat(enc2, dec3, 1))
        dec1 = self.dec1(torch.concat(enc1, dec2, 1))

        return self.final(dec1)
