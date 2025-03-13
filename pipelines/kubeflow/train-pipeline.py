"""
Kubeflow Pipelinesを使用したモデル学習パイプラインの定義
"""

import os
import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath, InputBool, InputInt, InputFloat
from typing import NamedTuple, Dict, List
import yaml


# データ前処理コンポーネント
@func_to_container_op
def preprocess_data(
    raw_data_path: InputPath(),
    processed_data_path: OutputPath(),
    test_size: float = 0.2,
    random_state: int = 42
) -> NamedTuple('PreprocessOutput', [('num_samples', int), ('num_features', int)]):
    """
    データの前処理を行い、訓練データとテストデータに分割する
    
    Args:
        raw_data_path: 生データのパス
        processed_data_path: 処理済みデータの出力パス
        test_size: テストデータの割合
        random_state: 分割の再現性のためのシード値
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    import os
    
    # データの読み込み
    df = pd.read_csv(raw_data_path)
    
    # NaNの処理
    df = df.fillna(df.mean())
    
    # 特徴量とターゲットの分離
    X = df.drop('target', axis=1)
    y = df['target']
    
    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 出力ディレクトリの作成
    os.makedirs(processed_data_path, exist_ok=True)
    
    # 処理済みデータの保存
    train_dir = os.path.join(processed_data_path, 'train')
    test_dir = os.path.join(processed_data_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(train_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(test_dir, 'X_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)
    
    # スケーラーの保存
    import pickle
    with open(os.path.join(processed_data_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 特徴量のリストを保存
    with open(os.path.join(processed_data_path, 'features.txt'), 'w') as f:
        f.write('\n'.join(X.columns))
    
    # 前処理の設定を保存
    with open(os.path.join(processed_data_path, 'preprocess_config.yaml'), 'w') as f:
        yaml.dump({
            'test_size': test_size,
            'random_state': random_state,
            'num_features': X.shape[1],
            'num_samples': len(df),
            'feature_names': X.columns.tolist()
        }, f)
    
    # 出力の生成
    PreprocessOutput = namedtuple('PreprocessOutput', ['num_samples', 'num_features'])
    return PreprocessOutput(len(df), X.shape[1])


# モデル学習コンポーネント
@func_to_container_op
def train_model(
    processed_data_path: InputPath(),
    model_path: OutputPath(),
    model_type: str = 'random_forest',
    hyperparams: str = '{}',
    mlflow_tracking_uri: str = '',
    experiment_name: str = 'kubeflow-training'
) -> NamedTuple('TrainOutput', [
    ('accuracy', float), 
    ('precision', float), 
    ('recall', float), 
    ('f1_score', float), 
    ('auc', float)
]):
    """
    モデルの学習を行う
    
    Args:
        processed_data_path: 処理済みデータのパス
        model_path: 学習済みモデルの出力パス
        model_type: モデルタイプ ('random_forest', 'xgboost', 'lightgbm')
        hyperparams: モデルのハイパーパラメータ (JSON文字列)
        mlflow_tracking_uri: MLflowのトラッキングURI
        experiment_name: MLflowの実験名
    """
    import pandas as pd
    import numpy as np
    import pickle
    import json
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    from collections import namedtuple
    import mlflow
    import mlflow.sklearn
    
    # MLflowの設定
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # ハイパーパラメータの解析
    hyperparams_dict = json.loads(hyperparams)
    
    # データの読み込み
    train_dir = os.path.join(processed_data_path, 'train')
    test_dir = os.path.join(processed_data_path, 'test')
    
    X_train = pd.read_csv(os.path.join(train_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(test_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(train_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv')).values.ravel()
    
    # MLflowの実行を開始
    with mlflow.start_run() as run:
        # 実行パラメータの記録
        mlflow.log_param('model_type', model_type)
        for key, value in hyperparams_dict.items():
            mlflow.log_param(key, value)
        
        # モデルの初期化と学習
        if model_type == 'random_forest':
            model = RandomForestClassifier(**hyperparams_dict)
        elif model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(**hyperparams_dict)
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**hyperparams_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # モデルの学習
        model.fit(X_train, y_train)
        
        # テストデータでの予測
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # 評価指標のログ
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('auc', auc)
        
        # 特徴量重要度のログ（モデルが対応している場合）
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = os.path.join(model_path, 'feature_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
        
        # 混同行列の保存
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        cm_path = os.path.join(model_path, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # ROC曲線の保存
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        roc_path = os.path.join(model_path, 'roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)
        
        # モデルの保存
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # MLflowにモデルを記録
        mlflow.sklearn.log_model(model, "model")
        
        # モデルメタデータの保存
        model_metadata = {
            'model_type': model_type,
            'hyperparameters': hyperparams_dict,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            },
            'mlflow_run_id': run.info.run_id,
            'mlflow_experiment_id': run.info.experiment_id
        }
        
        with open(os.path.join(model_path, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
    
    # 出力の生成
    TrainOutput = namedtuple('TrainOutput', ['accuracy', 'precision', 'recall', 'f1_score', 'auc'])
    return TrainOutput(accuracy, precision, recall, f1, auc)


# モデル評価コンポーネント
@func_to_container_op
def evaluate_model(
    model_path: InputPath(),
    evaluation_path: OutputPath(),
    processed_data_path: InputPath(),
    evaluation_metric: str = 'f1_score'
) -> NamedTuple('EvalOutput', [('metric_value', float), ('model_approved', bool)]):
    """
    モデルの詳細な評価を行う
    
    Args:
        model_path: 学習済みモデルのパス
        evaluation_path: 評価結果の出力パス
        processed_data_path: 処理済みデータのパス
        evaluation_metric: 主要評価指標
    """
    import pandas as pd
    import numpy as np
    import json
    import pickle
    import os
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        precision_recall_curve, average_precision_score
    )
    import matplotlib.pyplot as plt
    from collections import namedtuple
    
    # 出力ディレクトリの作成
    os.makedirs(evaluation_path, exist_ok=True)
    
    # モデルとメタデータの読み込み
    with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(model_path, 'model_metadata.json'), 'r') as f:
        model_metadata = json.load(f)
    
    # テストデータの読み込み
    test_dir = os.path.join(processed_data_path, 'test')
    X_test = pd.read_csv(os.path.join(test_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv')).values.ravel()
    
    # テストデータでの予測
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 詳細な評価指標の計算
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # レポートの保存
    with open(os.path.join(evaluation_path, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # PR曲線の作成
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    
    pr_curve_path = os.path.join(evaluation_path, 'pr_curve.png')
    plt.savefig(pr_curve_path)
    plt.close()
    
    # 決定境界のしきい値の検討
    thresholds_df = pd.DataFrame({
        'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'precision': [precision_score(y_test, y_prob >= t) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        'recall': [recall_score(y_test, y_prob >= t) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        'f1_score': [f1_score(y_test, y_prob >= t) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    })
    
    thresholds_df.to_csv(os.path.join(evaluation_path, 'threshold_analysis.csv'), index=False)
    
    # しきい値分析の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_df['threshold'], thresholds_df['precision'], label='Precision')
    plt.plot(thresholds_df['threshold'], thresholds_df['recall'], label='Recall')
    plt.plot(thresholds_df['threshold'], thresholds_df['f1_score'], label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()
    plt.grid(True)
    
    threshold_path = os.path.join(evaluation_path, 'threshold_analysis.png')
    plt.savefig(threshold_path)
    plt.close()
    
    # 評価の判断基準とモデル承認の決定
    target_metric_value = report['weighted avg'][evaluation_metric]
    threshold = 0.75  # 指標の閾値（調整可能）
    model_approved = target_metric_value >= threshold
    
    # 評価結果のまとめ
    evaluation_summary = {
        'evaluation_metric': evaluation_metric,
        'metric_value': target_metric_value,
        'threshold': threshold,
        'model_approved': model_approved,
        'classification_report': report,
        'average_precision': average_precision
    }
    
    with open(os.path.join(evaluation_path, 'evaluation_summary.json'), 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # 出力の生成
    EvalOutput = namedtuple('EvalOutput', ['metric_value', 'model_approved'])
    return EvalOutput(target_metric_value, model_approved)


# モデルパッケージングコンポーネント
@func_to_container_op
def package_model(
    model_path: InputPath(),
    package_path: OutputPath(),
    evaluation_path: InputPath(),
    model_approved: bool = True,
    version: str = '1.0.0'
) -> str:
    """
    モデルをデプロイ可能な形式にパッケージング
    
    Args:
        model_path: 学習済みモデルのパス
        package_path: パッケージングされたモデルの出力パス
        evaluation_path: 評価結果のパス
        model_approved: モデルが承認されたかどうか
        version: モデルのバージョン
    """
    import os
    import shutil
    import json
    import pickle
    import datetime
    
    # モデルが承認されない場合は処理をスキップ
    if not model_approved:
        print("Model not approved. Skipping packaging step.")
        return "Model not approved"
    
    # 出力ディレクトリの作成
    os.makedirs(package_path, exist_ok=True)
    
    # モデルファイルをコピー
    shutil.copy(os.path.join(model_path, 'model.pkl'), os.path.join(package_path, 'model.pkl'))
    
    # メタデータファイルの読み込みと更新
    with open(os.path.join(model_path, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # 評価サマリーの読み込み
    with open(os.path.join(evaluation_path, 'evaluation_summary.json'), 'r') as f:
        evaluation = json.load(f)
    
    # パッケージングメタデータの作成
    package_metadata = {
        **metadata,
        'version': version,
        'packaging_date': datetime.datetime.now().isoformat(),
        'evaluation': evaluation
    }
    
    # メタデータの保存
    with open(os.path.join(package_path, 'package_metadata.json'), 'w') as f:
        json.dump(package_metadata, f, indent=2)
    
    # 推論用のスクリプトを作成
    inference_script = """
import pickle
import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, data):
        \"\"\"
        データに対して予測を行う
        
        Args:
            data (pd.DataFrame): 特徴量を含むデータフレーム
            
        Returns:
            predictions: 予測クラスのリスト
            probabilities: 予測確率のリスト
        \"\"\"
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1]
        return predictions, probabilities

# 使用例
# predictor = ModelPredictor('model.pkl')
# predictions, probabilities = predictor.predict(df)
"""
    
    with open(os.path.join(package_path, 'predictor.py'), 'w') as f:
        f.write(inference_script)
    
    # READMEファイルの作成
    readme = f"""# Model Package v{version}

This package contains a trained machine learning model for prediction.

## Model Information
- Type: {metadata['model_type']}
- Version: {version}
- Packaging Date: {datetime.datetime.now().strftime('%Y-%m-%d')}

## Metrics
- Accuracy: {metadata['metrics']['accuracy']:.4f}
- Precision: {metadata['metrics']['precision']:.4f}
- Recall: {metadata['metrics']['recall']:.4f}
- F1 Score: {metadata['metrics']['f1_score']:.4f}
- AUC: {metadata['metrics']['auc']:.4f}

## Usage
```python
from predictor import ModelPredictor

predictor = ModelPredictor('model.pkl')
predictions, probabilities = predictor.predict(df)
```
"""
    
    with open(os.path.join(package_path, 'README.md'), 'w') as f:
        f.write(readme)
    
    return f"Model successfully packaged as version {version}"


# KubeflowパイプラインのDAG定義
@dsl.pipeline(
    name='Model Training Pipeline',
    description='End-to-end pipeline for training and evaluating ML models'
)
def training_pipeline(
    raw_data_url: str = 'gs://ml-data/raw_data.csv',
    mlflow_tracking_uri: str = 'http://mlflow-service.kubeflow:5000',
    model_type: str = 'random_forest',
    hyperparams: str = '{"n_estimators": 100, "max_depth": 10, "random_state": 42}',
    test_size: float = 0.2,
    evaluation_metric: str = 'f1_score',
    model_version: str = '1.0.0'
):
    """
    Kubeflow Pipelineのメイン定義
    
    Args:
        raw_data_url: 生データのGCSパス
        mlflow_tracking_uri: MLflowのトラッキングURI
        model_type: モデルのタイプ
        hyperparams: ハイパーパラメータのJSON文字列
        test_size: テストデータの割合
        evaluation_metric: 評価指標
        model_version: モデルバージョン
    """
    # ボリュームの設定
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="pipeline-pvc",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWO
    )
    
    # データのダウンロード
    download = dsl.ContainerOp(
        name='download_data',
        image='google/cloud-sdk:slim',
        command=['sh', '-c'],
        arguments=[f'gsutil cp {raw_data_url} /data/raw_data.csv'],
        pvolumes={"/data": vop.volume}
    )
    
    # データの前処理
    preprocess = preprocess_data(
        raw_data_path='/data/raw_data.csv',
        processed_data_path='/data/processed',
        test_size=test_size,
        random_state=42
    ).after(download).set_pvolumes({"/data": vop.volume})
    
    # モデルの学習
    train = train_model(
        processed_data_path='/data/processed',
        model_path='/data/model',
        model_type=model_type,
        hyperparams=hyperparams,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name='kubeflow-training'
    ).after(preprocess).set_pvolumes({"/data": vop.volume})
    
    # モデルの評価
    evaluate = evaluate_model(
        model_path='/data/model',
        evaluation_path='/data/evaluation',
        processed_data_path='/data/processed',
        evaluation_metric=evaluation_metric
    ).after(train).set_pvolumes({"/data": vop.volume})
    
    # モデルのパッケージング (評価結果に基づく条件付き実行)
    with dsl.Condition(evaluate.outputs['model_approved'] == 'true'):
        package = package_model(
            model_path='/data/model',
            package_path='/data/package',
            evaluation_path='/data/evaluation',
            model_approved=True,
            version=model_version
        ).set_pvolumes({"/data": vop.volume})
    
    # リソース制約の設定
    for op in [preprocess, train, evaluate]:
        op.set_memory_request('2G')
        op.set_memory_limit('4G')
        op.set_cpu_request('1')
        op.set_cpu_limit('2')


# パイプラインのコンパイル
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path='training_pipeline.yaml'
    )
