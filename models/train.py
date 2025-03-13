#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
モデル学習用のメインスクリプト
"""

import os
import argparse
import logging
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from dataloader import CustomDataset
from model_architecture import SimpleModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def train(config):
    """
    モデルの学習を実行する関数
    
    Args:
        config: 学習パラメータを含む設定辞書
    """
    # MLflowの実験を設定
    mlflow.set_experiment(config['experiment_name'])
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # データセットの準備
    train_dataset = CustomDataset(
        data_path=config['train_data_path'],
        transform=config['transforms']
    )
    val_dataset = CustomDataset(
        data_path=config['val_data_path'],
        transform=config['transforms']
    )
    
    # データローダーの設定
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # モデルの初期化
    model = SimpleModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size']
    )
    model = model.to(device)
    
    # 損失関数と最適化アルゴリズムの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 学習率スケジューラの設定
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5, 
        verbose=True
    )
    
    # MLflowでパラメータをログ
    with mlflow.start_run():
        # パラメータのログ
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
        
        # モデルアーキテクチャのログ
        mlflow.log_param("model_architecture", model.__class__.__name__)
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # 学習開始
        for epoch in range(config['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
            
            # 訓練モード
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 勾配をゼロに初期化
                optimizer.zero_grad()
                
                # 順伝播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 逆伝播と最適化
                loss.backward()
                optimizer.step()
                
                # 統計情報の更新
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if (i + 1) % config['log_interval'] == 0:
                    logger.info(f"Batch {i+1}/{len(train_loader)}: "
                                f"Loss: {train_loss/(i+1):.4f} | "
                                f"Acc: {100.*correct/total:.2f}%")
            
            # エポック終了時の訓練統計
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            
            # MLflowにメトリクスをログ
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            
            # 検証モード
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            # エポック終了時の検証統計
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * correct / total
            
            # MLflowにメトリクスをログ
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
            logger.info(f"Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_accuracy:.2f}% | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_accuracy:.2f}%")
            
            # 学習率の調整
            scheduler.step(val_loss)
            
            # ベストモデルの保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                
                # モデルの保存
                checkpoint_path = os.path.join(config['model_dir'], 'model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, checkpoint_path)
                
                # MLflowにモデルを記録
                mlflow.pytorch.log_model(model, "model")
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                early_stop_counter += 1
                
            # 早期停止
            if early_stop_counter >= config['early_stop_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training completed!")
        
        # メタデータの記録
        mlflow.set_tag("mlflow.runName", f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="Train a PyTorch model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    
    # 設定ファイルの読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # モデル保存ディレクトリの作成
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # 学習の実行
    train(config)


if __name__ == "__main__":
    main()
