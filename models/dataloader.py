#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
データローダーとデータ前処理モジュール
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Callable, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def preprocess_input(x: np.ndarray) -> np.ndarray:
    """
    モデル入力の標準的な前処理関数
    
    Args:
        x: 入力データ
        
    Returns:
        前処理済みデータ
    """
    # 特徴量のスケーリング（0〜1の範囲に正規化）
    x = x.astype(np.float32) / 255.0
    return x


class CustomDataset(Dataset):
    """
    カスタムデータセットクラス
    
    Args:
        data_path: データファイルパス（CSVまたはディレクトリ）
        transform: データ変換関数
        target_column: ターゲット列名（CSVの場合）
        feature_columns: 特徴量列名のリスト（CSVの場合、Noneの場合はターゲット以外全て）
    """
    def __init__(self, 
                 data_path: str, 
                 transform: Optional[Callable] = None,
                 target_column: str = 'target',
                 feature_columns: Optional[List[str]] = None):
        
        self.data_path = data_path
        self.transform = transform
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        self.data = None
        self.targets = None
        self.file_list = None
        self.class_map = None
        
        # データタイプに応じた読み込み
        if os.path.isfile(data_path):
            if data_path.endswith('.csv'):
                self._load_csv_data()
            elif data_path.endswith('.npy'):
                self._load_numpy_data()
            elif data_path.endswith('.json'):
                self._load_json_data()
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        elif os.path.isdir(data_path):
            # ディレクトリの場合は画像分類タスクと仮定
            self._load_image_data()
        else:
            raise ValueError(f"Data path does not exist: {data_path}")
    
    def _load_csv_data(self):
        """CSVデータの読み込み"""
        logger.info(f"Loading CSV data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # ターゲット列の確認
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in CSV")
        
        # 特徴量列の設定
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        # データとターゲットの抽出
        self.data = df[self.feature_columns].values.astype(np.float32)
        self.targets = df[self.target_column].values
        
        logger.info(f"Loaded {len(self.data)} samples with {len(self.feature_columns)} features")
        
    def _load_numpy_data(self):
        """NumPy配列データの読み込み"""
        logger.info(f"Loading NumPy data from {self.data_path}")
        
        # ファイルパスからファイル名の拡張子前の部分を取得
        base_path = os.path.splitext(self.data_path)[0]
        
        # Xとyを別々のファイルから読み込むことを想定
        try:
            self.data = np.load(self.data_path)
            target_path = f"{base_path}_targets.npy"
            
            if os.path.exists(target_path):
                self.targets = np.load(target_path)
            else:
                logger.warning(f"Target file {target_path} not found. Assuming inference mode.")
                self.targets = np.zeros(len(self.data))  # ダミーターゲット
        except Exception as e:
            logger.error(f"Error loading NumPy data: {e}")
            raise
        
        logger.info(f"Loaded {len(self.data)} samples with shape {self.data.shape[1:]}")
    
    def _load_json_data(self):
        """JSONデータの読み込み"""
        logger.info(f"Loading JSON data from {self.data_path}")
        
        try:
            with open(self.data_path, 'r') as f:
                json_data = json.load(f)
            
            # JSONの構造に応じて処理を分岐
            if isinstance(json_data, list):
                # リスト形式のJSONを処理
                if all(isinstance(item, dict) for item in json_data):
                    # 辞書のリストを処理
                    df = pd.DataFrame(json_data)
                    
                    if self.target_column in df.columns:
                        self.targets = df[self.target_column].values
                        
                        if self.feature_columns is None:
                            self.feature_columns = [col for col in df.columns if col != self.target_column]
                        
                        self.data = df[self.feature_columns].values.astype(np.float32)
                    else:
                        # ターゲット列がない場合は推論モードと仮定
                        self.data = df.values.astype(np.float32)
                        self.targets = np.zeros(len(self.data))  # ダミーターゲット
                else:
                    # 単純なリストを処理
                    self.data = np.array(json_data, dtype=np.float32)
                    self.targets = np.zeros(len(self.data))  # ダミーターゲット
            else:
                # 辞書形式のJSONを処理
                if 'data' in json_data and 'target' in json_data:
                    self.data = np.array(json_data['data'], dtype=np.float32)
                    self.targets = np.array(json_data['target'])
                else:
                    raise ValueError("JSON format not recognized")
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def _load_image_data(self):
        """
        画像データの読み込み
        ディレクトリ構造は次の形式を想定:
        data_path/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── class2/
        │   ├── img1.jpg
        │   └── ...
        └── ...
        """
        logger.info(f"Loading image data from directory {self.data_path}")
        
        self.file_list = []
        self.targets = []
        
        # クラスディレクトリをリスト化
        class_dirs = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d))]
        
        # クラスマッピングの作成
        self.class_map = {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}
        
        # 画像ファイルの収集
        for class_name in class_dirs:
            class_dir = os.path.join(self.data_path, class_name)
            class_idx = self.class_map[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.file_list.append(os.path.join(class_dir, filename))
                    self.targets.append(class_idx)
        
        # デフォルトの変換がNoneの場合、標準的な画像変換を設定
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        
        logger.info(f"Loaded {len(self.file_list)} images from {len(class_dirs)} classes")
        
        # クラスマッピングをJSONファイルとして保存
        class_map_path = os.path.join(self.data_path, "class_mapping.json")
        with open(class_map_path, 'w') as f:
            json.dump({str(v): k for k, v in self.class_map.items()}, f, indent=2)
        
        logger.info(f"Saved class mapping to {class_map_path}")
    
    def __len__(self) -> int:
        """データセットの長さを返す"""
        if self.file_list is not None:
            return len(self.file_list)
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        インデックスに対応するサンプルを返す
        
        Args:
            idx: サンプルのインデックス
            
        Returns:
            (特徴量, ターゲット)のタプル
        """
        if self.file_list is not None:
            # 画像データセットの場合
            img_path = self.file_list[idx]
            target = self.targets[idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                return image, target
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                # エラー時に代替画像（黒画像）を返す
                return torch.zeros(3, 224, 224), target
        else:
            # テーブルデータの場合
            features = self.data[idx]
            target = self.targets[idx]
            
            if self.transform:
                features = self.transform(features)
            
            return torch.FloatTensor(features), target


def create_data_loaders(
    train_path: str,
    val_path: str = None,
    test_path: str = None,
    batch_size: int = 32,
    transform: Optional[Callable] = None,
    target_column: str = 'target',
    feature_columns: Optional[List[str]] = None,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
) -> Dict[str, DataLoader]:
    """
    訓練・検証・テスト用のデータローダーを作成
    
    Args:
        train_path: 訓練データのパス
        val_path: 検証データのパス（指定されない場合はtrain_pathから分割）
        test_path: テストデータのパス（指定されない場合はtrain_pathから分割）
        batch_size: バッチサイズ
        transform: データ変換関数
        target_column: ターゲット列名
        feature_columns: 特徴量列名のリスト
        num_workers: データロード並列ワーカー数
        val_ratio: 検証データの割合（val_pathが指定されない場合）
        test_ratio: テストデータの割合（test_pathが指定されない場合）
        
    Returns:
        データローダーの辞書 {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    from torch.utils.data import random_split
    
    loaders = {}
    
    # 訓練データセットの作成
    full_dataset = CustomDataset(
        data_path=train_path,
        transform=transform,
        target_column=target_column,
        feature_columns=feature_columns
    )
    
    # 外部のval_pathとtest_pathが指定されていない場合はデータセットを分割
    if val_path is None or test_path is None:
        total_size = len(full_dataset)
        val_size = int(total_size * val_ratio)
        test_size = int(total_size * test_ratio)
        train_size = total_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # 外部のパスが指定されている場合は個別にデータセットを作成
        loaders['train'] = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        if val_path:
            val_dataset = CustomDataset(
                data_path=val_path,
                transform=transform,
                target_column=target_column,
                feature_columns=feature_columns
            )
            
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        if test_path:
            test_dataset = CustomDataset(
                data_path=test_path,
                transform=transform,
                target_column=target_column,
                feature_columns=feature_columns
            )
            
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    return loaders


# データ拡張の定義例
def get_image_transforms(train=True):
    """
    画像データの変換関数を取得
    
    Args:
        train: 訓練用の変換を返すかどうか
        
    Returns:
        変換関数
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
