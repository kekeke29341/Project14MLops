#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
モデルアーキテクチャを定義するモジュール
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """
    シンプルなニューラルネットワークモデル
    
    Args:
        input_size (int): 入力特徴量の次元数
        hidden_size (int): 隠れ層のユニット数
        output_size (int): 出力クラス数
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        """
        順伝播処理
        
        Args:
            x (torch.Tensor): 入力テンソル [batch_size, input_size]
            
        Returns:
            torch.Tensor: 出力テンソル [batch_size, output_size]
        """
        # 第1層
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 第2層
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 出力層
        x = self.fc3(x)
        
        return x


class ConvolutionalModel(nn.Module):
    """
    畳み込みニューラルネットワークモデル（画像分類用）
    
    Args:
        num_classes (int): 出力クラス数
    """
    def __init__(self, num_classes=10):
        super(ConvolutionalModel, self).__init__()
        
        # 畳み込み層ブロック1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # 畳み込み層ブロック2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        
        # 畳み込み層ブロック3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4)
        
        # グローバルプーリング
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全結合層
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        順伝播処理
        
        Args:
            x (torch.Tensor): 入力画像テンソル [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: 出力テンソル [batch_size, num_classes]
        """
        # ブロック1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # ブロック2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # ブロック3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # グローバルプーリングと全結合層
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x


# 追加のモデルアーキテクチャを必要に応じて定義
