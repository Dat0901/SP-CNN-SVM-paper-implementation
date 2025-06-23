import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=None, 
                 pool_kernel=2, 
                 pool_stride=2, 
                 pool=True, 
                 bn=True):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2
        if padding < 0:
            raise ValueError("Padding must be non-negative")
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)) # Try ernel_size=3, stride=2 later
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__() 
        # Features (why code like this cuz paper need the output of conv5, fc6, fc7 as features for SVM)
        self.features = nn.Sequential(
            ConvBlock(3, 96, kernel_size=11, stride=4, padding=0, pool=True, pool_kernel=2, pool_stride=2, bn=True),
            ConvBlock(96, 256, kernel_size=5, stride=1, padding=2, pool=True, pool_kernel=2, pool_stride=2, bn=True),
            ConvBlock(256, 384, kernel_size=3, stride=1, padding=1, pool=False, bn=False),
            ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, pool=False, bn=False),
        )
        self.conv5 = ConvBlock(384, 256, kernel_size=3, stride=1, padding=1, pool=True, pool_kernel=2, pool_stride=2, bn=True)

        # Fully connected layers
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),  # Assuming input size is 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x, return_features=False):
        """
        returns features so that we can use them for SVM
        """
        x = self.features(x)
        conv5_feat = self.conv5(x)
        x_flat = conv5_feat.view(conv5_feat.size(0), -1)
        fc6_feat = self.fc6(x_flat)
        fc7_feat = self.fc7(fc6_feat)
        out = self.fc8(fc7_feat)

        if return_features:
            return conv5_feat, fc6_feat, fc7_feat
        return out
