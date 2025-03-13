#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
モデル推論用スクリプト
"""

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_architecture import SimpleModel, ConvolutionalModel
from dataloader import CustomDataset, preprocess_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class ModelInference:
    """
    モデル推論を行うクラス
    """
    def __init__(self, model_path: str, config_path: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_path: モデルの重みファイルへのパス
            config_path: モデル設定ファイルへのパス
            device: 推論に使用するデバイス ('cuda' または 'cpu')
        """
        self.config = self._load_config(config_path)
        
        # デバイスの設定
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # モデルの初期化と重みのロード
        self.model = self._load_model(model_path)
        
        # クラスマッピングの読み込み（存在する場合）
        self.class_mapping = self._load_class_mapping()
        
        logger.info("Model loaded and ready for inference")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        モデル設定を読み込み
        
        Args:
            config_path: 設定ファイルへのパス
            
        Returns:
            設定辞書
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        モデルを読み込み
        
        Args:
            model_path: モデルの重みファイルへのパス
            
        Returns:
            ロードされたモデル
        """
        # モデルアーキテクチャの選択
        model_type = self.config.get('model_type', 'SimpleModel')
        
        if model_type == 'SimpleModel':
            model = SimpleModel(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                output_size=self.config['output_size']
            )
        elif model_type == 'ConvolutionalModel':
            model = ConvolutionalModel(
                num_classes=self.config['output_size']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # モデルの重みをロード
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = torch.jit.load(model_path, map_location=self.device)
        
        model = model.to(self.device)
        model.eval()  # 推論モードに設定
        
        return model
    
    def _load_class_mapping(self) -> Dict[int, str]:
        """
        クラスIDとラベルのマッピングを読み込み
        
        Returns:
            クラスマッピング辞書
        """
        class_mapping_path = self.config.get('class_mapping_path')
        if not class_mapping_path or not os.path.exists(class_mapping_path):
            return {i: str(i) for i in range(self.config['output_size'])}
        
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        # キーを整数型に変換
        return {int(k): v for k, v in class_mapping.items()}
    
    def predict_one(self, input_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        単一の入力データに対する予測を行う
        
        Args:
            input_data: 入力データ
            
        Returns:
            予測結果を含む辞書
        """
        start_time = time.time()
        
        # データの前処理
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data.float()
        
        # バッチ次元の追加
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # 推論
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 結果の整形
        inference_time = time.time() - start_time
        
        result = {
            "predicted_class_id": predicted_class,
            "predicted_class_name": self.class_mapping.get(predicted_class, str(predicted_class)),
            "confidence": confidence,
            "inference_time_ms": inference_time * 1000
        }
        
        return result
    
    def predict_batch(self, data_loader: DataLoader) -> List[Dict[str, Any]]:
        """
        バッチデータに対する予測を行う
        
        Args:
            data_loader: データローダー
            
        Returns:
            予測結果のリスト
        """
        results = []
        start_time = time.time()
        
        for batch_data in data_loader:
            # データロードの形式に応じて処理を変更
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                inputs, _ = batch_data  # ラベル情報は無視
            else:
                inputs = batch_data
            
            inputs = inputs.to(self.device)
            
            # 推論
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                confidences = torch.gather(probabilities, 1, predicted_classes.unsqueeze(1))
            
            # バッチ内の各サンプルに対する結果を追加
            for i in range(inputs.size(0)):
                predicted_class = predicted_classes[i].item()
                result = {
                    "predicted_class_id": predicted_class,
                    "predicted_class_name": self.class_mapping.get(predicted_class, str(predicted_class)),
                    "confidence": confidences[i].item(),
                }
                results.append(result)
        
        batch_inference_time = time.time() - start_time
        logger.info(f"Batch inference completed in {batch_inference_time:.2f} seconds")
        
        return results
    
    def export_onnx(self, output_path: str, input_shape: List[int]) -> None:
        """
        ONNXフォーマットでモデルをエクスポート
        
        Args:
            output_path: 出力ONNXファイルパス
            input_shape: 入力テンソルの形状
        """
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # TorchScriptモデルはONNXエクスポートに直接使用できる
        if isinstance(self.model, torch.jit.ScriptModule):
            model = self.model
        else:
            model = self.model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX format at {output_path}")


def create_inference_server(model_path, config_path, host="0.0.0.0", port=8000):
    """
    FastAPIを使用した推論サーバーの作成（実際の実装時にはこの関数を展開）
    
    Args:
        model_path: モデルパス
        config_path: 設定ファイルパス
        host: サーバーホスト
        port: サーバーポート
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="Model Inference API")
        
        # モデル推論インスタンスの作成
        inference = ModelInference(model_path, config_path)
        
        # 入力データモデル
        class InputData(BaseModel):
            data: List[List[float]]
        
        # 推論エンドポイント
        @app.post("/predict")
        async def predict(input_data: InputData):
            try:
                input_array = np.array(input_data.data, dtype=np.float32)
                result = inference.predict_one(input_array)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # サーバー起動
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("FastAPI or uvicorn not installed. Please install with: pip install fastapi uvicorn")


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--input", type=str, help="Path to the input data file")
    parser.add_argument("--output", type=str, help="Path to save inference results")
    parser.add_argument("--batch", action="store_true", help="Run inference in batch mode")
    parser.add_argument("--server", action="store_true", help="Start an inference server")
    parser.add_argument("--export-onnx", type=str, help="Export model to ONNX format")
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    
    # モデル推論インスタンスの作成
    inference = ModelInference(args.model, args.config)
    
    if args.server:
        create_inference_server(args.model, args.config)
    elif args.export_onnx:
        # モデル設定から入力形状を取得
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # 入力形状の設定
        if config.get('model_type') == 'ConvolutionalModel':
            # 畳み込みモデルの場合は画像入力を想定
            input_shape = [1, 3, 224, 224]  # バッチサイズ, チャネル数, 高さ, 幅
        else:
            # その他のモデルの場合は特徴ベクトルを想定
            input_shape = [1, config.get('input_size', 784)]  # バッチサイズ, 特徴数
        
        inference.export_onnx(args.export_onnx, input_shape)
    elif args.batch and args.input:
        # バッチ推論モード
        dataset = CustomDataset(args.input, transform=preprocess_input)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        results = inference.predict_batch(dataloader)
        
        # 結果の保存
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch inference results saved to {args.output}")
    elif args.input:
        # 単一データの推論
        try:
            # テキストファイルかCSVから読み込み
            input_data = np.loadtxt(args.input, delimiter=',')
            result = inference.predict_one(input_data)
            
            logger.info(f"Prediction: {result}")
            
            # 結果の保存
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Inference result saved to {args.output}")
                
        except Exception as e:
            logger.error(f"Error processing input file: {e}")
    else:
        logger.error("Either --input or --server or --export-onnx option must be specified")


if __name__ == "__main__":
    main()
