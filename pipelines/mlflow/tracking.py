"""
MLflowのトラッキングとモデル管理のためのユーティリティ関数
"""

import os
import json
import pickle
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflowを使用してモデル学習と評価を追跡するクラス
    """
    def __init__(self, 
                experiment_name: str, 
                tracking_uri: Optional[str] = None,
                artifact_uri: Optional[str] = None):
        """
        MLflowトラッキングの初期化
        
        Args:
            experiment_name: MLflow実験名
            tracking_uri: MLflowトラッキングサーバーURI
            artifact_uri: アーティファクト保存先URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_uri = artifact_uri
        
        # MLflowの設定
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # クライアントの初期化
        self.client = MlflowClient(tracking_uri)
        
        # 実験の設定
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                # 存在しない場合は新しい実験を作成
                experiment_id = mlflow.create_experiment(
                    experiment_name, 
                    artifact_location=artifact_uri
                )
                self.experiment = mlflow.get_experiment(experiment_id)
            
            self.experiment_id = self.experiment.experiment_id
            logger.info(f"実験 '{experiment_name}' を準備しました（ID: {self.experiment_id}）")
        
        except Exception as e:
            logger.error(f"MLflow実験の設定中にエラーが発生しました: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        MLflowの実行を開始
        
        Args:
            run_name: 実行名（省略可）
            
        Returns:
            MLflowのアクティブな実行
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def log_model_training(self, 
                          model: Any, 
                          model_params: Dict[str, Any],
                          metrics: Dict[str, float],
                          features: pd.DataFrame,
                          feature_importance: Optional[Dict[str, float]] = None,
                          model_name: str = "model",
                          model_description: Optional[str] = None,
                          tags: Optional[Dict[str, str]] = None) -> str:
        """
        モデル学習に関する情報をログ
        
        Args:
            model: 学習済みモデル
            model_params: モデルのパラメータ
            metrics: 評価指標の辞書
            features: 特徴量のDataFrame（特徴量名を取得するため）
            feature_importance: 特徴量重要度の辞書（省略可）
            model_name: モデル名
            model_description: モデルの説明
            tags: 追加のタグ
            
        Returns:
            MLflowの実行ID
        """
        with self.start_run(run_name=f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}") as run:
            # パラメータのログ
            for key, value in model_params.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
                else:
                    # 複雑な型はJSON文字列に変換
                    mlflow.log_param(key, str(value))
            
            # メトリクスのログ
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # タグの設定
            all_tags = {
                "model_type": type(model).__name__,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            
            if model_description:
                all_tags["description"] = model_description
            
            if tags:
                all_tags.update(tags)
            
            for key, value in all_tags.items():
                mlflow.set_tag(key, value)
            
            # 特徴量重要度のログ（提供されている場合）
            if feature_importance or hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                
                if feature_importance:
                    importance_df = pd.DataFrame({
                        'feature': list(feature_importance.keys()),
                        'importance': list(feature_importance.values())
                    })
                elif hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': features.columns,
                        'importance': model.feature_importances_
                    })
                
                # 重要度でソート
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # 上位20件に制限（大量の特徴量がある場合）
                if len(importance_df) > 20:
                    importance_df = importance_df.head(20)
                
                # プロット
                sns.barplot(x='importance', y='feature', data=importance_df)
                plt.title('特徴量重要度')
                plt.tight_layout()
                
                # 一時ファイルに保存
                feature_importance_path = f"/tmp/feature_importance_{run.info.run_id}.png"
                plt.savefig(feature_importance_path)
                plt.close()
                
                # MLflowにログ
                mlflow.log_artifact(feature_importance_path)
                os.remove(feature_importance_path)  # 一時ファイルの削除
                
                # CSVとしても保存
                importance_csv_path = f"/tmp/feature_importance_{run.info.run_id}.csv"
                importance_df.to_csv(importance_csv_path, index=False)
                mlflow.log_artifact(importance_csv_path)
                os.remove(importance_csv_path)  # 一時ファイルの削除
            
            # モデルのログ
            mlflow.sklearn.log_model(model, model_name)
            
            # モデルをファイルとしても保存（省略可）
            model_path = f"/tmp/{model_name}_{run.info.run_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)
            os.remove(model_path)  # 一時ファイルの削除
            
            return run.info.run_id
    
    def log_evaluation_results(self, 
                              run_id: str,
                              eval_metrics: Dict[str, float],
                              confusion_matrix: Optional[np.ndarray] = None,
                              classification_report: Optional[Dict[str, Any]] = None,
                              roc_curve_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> None:
        """
        モデル評価結果をログ
        
        Args:
            run_id: MLflowの実行ID
            eval_metrics: 評価指標の辞書
            confusion_matrix: 混同行列（省略可）
            classification_report: 分類レポート（省略可）
            roc_curve_data: ROC曲線データ（fpr, tpr, thresholds）（省略可）
        """
        with mlflow.start_run(run_id=run_id):
            # 評価指標のログ
            for key, value in eval_metrics.items():
                mlflow.log_metric(f"eval_{key}", value)
            
            # 混同行列のログ（提供されている場合）
            if confusion_matrix is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('予測クラス')
                plt.ylabel('実際のクラス')
                plt.title('混同行列')
                
                # 一時ファイルに保存
                cm_path = f"/tmp/confusion_matrix_{run_id}.png"
                plt.savefig(cm_path)
                plt.close()
                
                # MLflowにログ
                mlflow.log_artifact(cm_path)
                os.remove(cm_path)  # 一時ファイルの削除
            
            # 分類レポートのログ（提供されている場合）
            if classification_report is not None:
                report_path = f"/tmp/classification_report_{run_id}.json"
                with open(report_path, 'w') as f:
                    json.dump(classification_report, f, indent=2)
                
                mlflow.log_artifact(report_path)
                os.remove(report_path)  # 一時ファイルの削除
            
            # ROC曲線のログ（提供されている場合）
            if roc_curve_data is not None:
                fpr, tpr, _ = roc_curve_data
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {eval_metrics.get("auc", 0):.3f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('偽陽性率 (FPR)')
                plt.ylabel('真陽性率 (TPR)')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend()
                
                # 一時ファイルに保存
                roc_path = f"/tmp/roc_curve_{run_id}.png"
                plt.savefig(roc_path)
                plt.close()
                
                # MLflowにログ
                mlflow.log_artifact(roc_path)
                os.remove(roc_path)  # 一時ファイルの削除
                
                # ROCデータをCSVとしても保存
                roc_data = pd.DataFrame({
                    'fpr': fpr,
                    'tpr': tpr
                })
                
                roc_data_path = f"/tmp/roc_data_{run_id}.csv"
                roc_data.to_csv(roc_data_path, index=False)
                mlflow.log_artifact(roc_data_path)
                os.remove(roc_data_path)  # 一時ファイルの削除
    
    def register_model(self, 
                       run_id: str, 
                       model_name: str,
                       description: Optional[str] = None,
                       tags: Optional[Dict[str, str]] = None) -> str:
        """
        モデルをMLflowモデルレジストリに登録
        
        Args:
            run_id: MLflowの実行ID
            model_name: 登録するモデル名
            description: モデルの説明
            tags: 追加のタグ
            
        Returns:
            モデルバージョン
        """
        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name
            )
            
            # 説明の追加
            if description:
                self.client.update_registered_model(
                    name=model_name,
                    description=description
                )
            
            # タグの追加
            if tags:
                for key, value in tags.items():
                    self.client.set_registered_model_tag(
                        name=model_name,
                        key=key,
                        value=value
                    )
            
            logger.info(f"モデル '{model_name}' をバージョン {result.version} として登録しました")
            return result.version
        
        except Exception as e:
            logger.error(f"モデル登録中にエラーが発生しました: {e}")
            raise
    
    def transition_model_stage(self, 
                              model_name: str, 
                              version: str, 
                              stage: str,
                              archive_existing_versions: bool = True) -> None:
        """
        モデルのステージを変更
        
        Args:
            model_name: モデル名
            version: モデルバージョン
            stage: 新しいステージ ('Staging', 'Production', 'Archived')
            archive_existing_versions: 同じステージの既存バージョンをアーカイブするかどうか
        """
        try:
            if archive_existing_versions and stage in ('Staging', 'Production'):
                # 同じステージの既存バージョンを検索
                existing_versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
                
                # 既存バージョンをアーカイブ
                for ev in existing_versions:
                    if ev.version != version:
                        logger.info(f"バージョン {ev.version} を 'Archived' に移行します")
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=ev.version,
                            stage="Archived"
                        )
            
            # 指定されたバージョンを新しいステージに移行
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"モデル '{model_name}' のバージョン {version} を '{stage}' ステージに移行しました")
            
        except Exception as e:
            logger.error(f"モデルステージの変更中にエラーが発生しました: {e}")
            raise
    
    def get_best_run(self, 
                    metric: str = 'accuracy', 
                    ascending: bool = False,
                    top_n: int = 1) -> List[Dict[str, Any]]:
        """
        指定されたメトリクスに基づいて最良のモデル実行を取得
        
        Args:
            metric: 評価する指標
            ascending: 昇順でソートするかどうか（Falseの場合は降順）
            top_n: 上位N件の結果を返す
            
        Returns:
            上位N件の実行情報のリスト
        """
        # 実験内のすべての実行を取得
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric} IS NOT NULL",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if runs.empty:
            logger.warning(f"メトリクス '{metric}' を持つ実行が見つかりませんでした")
            return []
        
        # 上位N件を選択
        top_runs = runs.head(top_n)
        
        # 結果をリストに変換
        result = []
        for _, row in top_runs.iterrows():
            run_info = {
                'run_id': row['run_id'],
                'experiment_id': row['experiment_id'],
                metric: row[f'metrics.{metric}'],
                'start_time': datetime.datetime.fromtimestamp(row['start_time'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'tags': {k.split('.')[-1]: v for k, v in row.items() if k.startswith('tags.')}
            }
            
            # 利用可能なすべてのメトリクスを追加
            for col in row.index:
                if col.startswith('metrics.'):
                    metric_name = col.split('.')[-1]
                    run_info[metric_name] = row[col]
            
            result.append(run_info)
        
        return result
    
    def compare_runs(self, 
                    run_ids: List[str], 
                    metrics: List[str] = None) -> pd.DataFrame:
        """
        複数の実行を比較
        
        Args:
            run_ids: 比較する実行IDのリスト
            metrics: 比較するメトリクスのリスト（省略時は全て）
            
        Returns:
            比較結果のDataFrame
        """
        # 実行データの取得
        comparison_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            
            run_data = {
                'run_id': run_id,
                'start_time': datetime.datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': run.data.tags.get('model_type', 'Unknown')
            }
            
            # パラメータの追加
            for key, value in run.data.params.items():
                run_data[f'param_{key}'] = value
            
            # メトリクスの追加（指定されている場合は指定されたもののみ）
            for key, value in run.data.metrics.items():
                if metrics is None or key in metrics:
                    run_data[f'metric_{key}'] = value
            
            comparison_data.append(run_data)
        
        # DataFrameに変換
        comparison_df = pd.DataFrame(comparison_data)
        
        # 列の並び替え
        cols = ['run_id', 'start_time', 'model_type']
        
        # まずメトリクス列を追加
        metric_cols = [col for col in comparison_df.columns if col.startswith('metric_')]
        metric_cols.sort()
        cols.extend(metric_cols)
        
        # 次にパラメータ列を追加
        param_cols = [col for col in comparison_df.columns if col.startswith('param_')]
        param_cols.sort()
        cols.extend(param_cols)
        
        # 他の列を追加
        other_cols = [col for col in comparison_df.columns if col not in cols]
        cols.extend(other_cols)
        
        return comparison_df[cols]
    
    def load_model(self, 
                  model_name: str, 
                  version: Optional[str] = None, 
                  stage: Optional[str] = 'Production') -> Any:
        """
        モデルレジストリからモデルをロード
        
        Args:
            model_name: モデル名
            version: モデルバージョン（省略時はステージから決定）
            stage: モデルステージ（version省略時のみ使用）
            
        Returns:
            ロードされたモデル
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"モデル '{model_uri}' をロードしました")
            return model
        except Exception as e:
            logger.error(f"モデルロード中にエラーが発生しました: {e}")
            raise


def setup_mlflow(tracking_uri: Optional[str] = None,
                artifact_location: Optional[str] = None) -> None:
    """
    MLflowの基本設定を行う
    
    Args:
        tracking_uri: MLflowトラッキングサーバーURI
        artifact_location: アーティファクト保存先URI
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflowトラッキングURIを {tracking_uri} に設定しました")
    
    # 環境変数にアーティファクト保存先を設定
    if artifact_location:
        os.environ['MLFLOW_ARTIFACT_ROOT'] = artifact_location
        logger.info(f"MLflowアーティファクト保存先を {artifact_location} に設定しました")


def find_best_model(experiment_name: str,
                   metric: str = 'accuracy',
                   ascending: bool = False,
                   tracking_uri: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    指定された実験から最良のモデルを見つける
    
    Args:
        experiment_name: 実験名
        metric: 評価指標
        ascending: 昇順または降順
        tracking_uri: MLflowトラッキングサーバーURI
        
    Returns:
        最良モデルの情報
    """
    # トラッカーの初期化
    tracker = MLflowTracker(experiment_name, tracking_uri)
    
    # 最良の実行を取得
    best_runs = tracker.get_best_run(metric, ascending)
    
    if not best_runs:
        logger.warning(f"実験 '{experiment_name}' で最良のモデルが見つかりませんでした")
        return None
    
    return best_runs[0]


def promote_model_to_production(model_name: str,
                               version: str,
                               tracking_uri: Optional[str] = None) -> None:
    """
    モデルをプロダクションステージに昇格
    
    Args:
        model_name: モデル名
        version: モデルバージョン
        tracking_uri: MLflowトラッキングサーバーURI
    """
    # MLflowクライアントの初期化
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    try:
        # 現在のプロダクションモデルを取得
        production_models = client.get_latest_versions(model_name, stages=["Production"])
        
        # 新しいバージョンをプロダクションに移行
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(f"モデル '{model_name}' バージョン {version} をプロダクションに昇格しました")
        
        # 以前のプロダクションモデルをアーカイブ
        for model in production_models:
            if model.version != version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model.version,
                    stage="Archived"
                )
                logger.info(f"以前のプロダクションモデル '{model_name}' バージョン {model.version} をアーカイブしました")
    
    except Exception as e:
        logger.error(f"モデル昇格中にエラーが発生しました: {e}")
        raise


# 使用例:
if __name__ == "__main__":
    # MLflowの設定
    setup_mlflow(
        tracking_uri="http://mlflow:5000",
        artifact_location="s3://mlflow-artifacts/models"
    )
    
    # トラッカーの初期化
    tracker = MLflowTracker(
        experiment_name="model_training",
        tracking_uri="http://mlflow:5000",
        artifact_uri="s3://mlflow-artifacts/models"
    )
    
    # サンプルモデルとメトリクス（実際には学習の結果）
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.79,
        'f1_score': 0.80,
        'auc': 0.90
    }
    
    # 特徴量のダミーデータ
    features = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })
    
    # モデル学習のログ
    run_id = tracker.log_model_training(
        model=model,
        model_params=params,
        metrics=metrics,
        features=features,
        model_name="random_forest_classifier",
        model_description="ランダムフォレスト分類器のサンプル",
        tags={"environment": "dev", "dataset": "sample"}
    )
    
    # モデルをレジストリに登録
    version = tracker.register_model(
        run_id=run_id,
        model_name="customer_churn_model",
        description="顧客離脱予測モデル"
    )
    
    # プロダクションに昇格
    tracker.transition_model_stage(
        model_name="customer_churn_model",
        version=version,
        stage="Production"
    )
    
    # 最良のモデルを見つける
    best_model = find_best_model(
        experiment_name="model_training",
        metric="f1_score"
    )
    
    if best_model:
        print(f"最良のモデル実行ID: {best_model['run_id']}")
        print(f"F1スコア: {best_model['f1_score']}")
