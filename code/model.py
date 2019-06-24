import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet34


class ChannelEvolver:
    def __init__(self, base_channels, ratio):
        self.base_channels = base_channels
        self.ratio = ratio

    def __call__(self, n):
        res = self.base_channels
        for i in range(0, n):
            res = int(res * self.ratio)
        return res


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout_prob=0., upsample=True, use_padding=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.ReplicationPad2d(1) if use_padding else nn.Sequential(),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1) if use_padding else nn.Sequential(),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob, inplace=False)
        )
        if upsample:
            self.decode.add_module('convTranspose',
                                   nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2))

    def forward(self, x):
        return self.decode(x)


class Model(nn.Module):
    def __init__(self, img_width, num_segmentation_classes, resnet_pre_trained=True, dropout_bottleneck=0.):
        super(Model, self).__init__()
        self.img_size = img_width
        # this is mandatory to avoid dimensions mismatch when ratio is not 2.0
        encoder_sizes = []
        # ENCODER
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))
        self.register_buffer('resnet_mean', torch.FloatTensor(mean))
        self.register_buffer('resnet_std', torch.FloatTensor(std))

        # HERE WE REPLACE SOME PARAMETERS ACCORDING TO RESNET STRUCTURE
        num_layers = 4
        base_channels = 64
        ratio = 2.0
        self.channel_ev = ChannelEvolver(base_channels=base_channels, ratio=ratio)
        r = resnet34(resnet_pre_trained)
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                r.conv1,
                r.bn1,
                r.relu,
                r.layer1),
            r.layer2,
            r.layer3, r.layer4
        ])
        for i in range(num_layers):
            encoder_sizes.append(self.channel_ev(i))

        # CENTRAL BLOCK
        i = num_layers
        self.center = _DecoderBlock(self.channel_ev(i - 1), self.channel_ev(i), self.channel_ev(i - 1),
                                    dropout_prob=dropout_bottleneck, use_padding=True)

        # DECODER
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            i = num_layers - i - 1
            if i == 0:
                b = _DecoderBlock(encoder_sizes[i] * 2, encoder_sizes[i], encoder_sizes[i],
                                  dropout_prob=0.5, upsample=False, use_padding=True)
            else:
                b = _DecoderBlock(encoder_sizes[i] * 2, encoder_sizes[i], encoder_sizes[i - 1],
                                  dropout_prob=0.5, use_padding=True)
            self.decoder_layers.append(b)
        # SEGMENTATION
        self.segmentation = nn.Conv2d(base_channels, num_segmentation_classes, kernel_size=1)

    def forward(self, x):
        enc_features = []
        feat = (x - self.resnet_mean) / self.resnet_std
        for b in self.encoder_layers:
            feat = b(feat)
            enc_features.append(feat)
        feat = self.center(feat)
        # DECODER
        for b, enc_feat in zip(self.decoder_layers, enc_features[::-1]):
            feat = b(torch.cat([feat, F.interpolate(enc_feat, feat.size()[2:], mode='bilinear', align_corners=False)], 1))

        segmentation = self.segmentation(feat)
        return F.interpolate(segmentation, x.size()[2:], mode='bilinear', align_corners=False)
