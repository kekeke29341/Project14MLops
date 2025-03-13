"""
機械学習モデルを学習してMLflowに記録するAirflow DAG
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.models import Variable

import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# デフォルト引数
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAGの定義
dag = DAG(
    'ml_model_training',
    default_args=default_args,
    description='機械学習モデルのトレーニングパイプライン',
    schedule_interval='30 2 * * *',  # ETLパイプラインの30分後に実行
    catchup=False,
)

# 環境変数とパスの設定
DATA_DIR = '/opt/airflow/data/processed'
MODEL_DIR = '/opt/airflow/models'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# MLflowの設定
MLFLOW_TRACKING_URI = Variable.get('mlflow_tracking_uri', default_var='http://mlflow:5000')
MLFLOW_EXPERIMENT_NAME = 'model_training'

# 必要なディレクトリの作成
for directory in [MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_training_data(**kwargs):
    """
    訓練データと検証データの読み込み
    """
    # 訓練データの読み込み
    X_train = pd.read_csv(os.path.join(TRAIN_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, 'y_train.csv')).values.ravel()
    
    # テストデータの読み込み
    X_test = pd.read_csv(os.path.join(TEST_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(TEST_DIR, 'y_test.csv')).values.ravel()
    
    # データの一部をXComに渡す（小規模の場合のみ）
    kwargs['ti'].xcom_push(key='feature_columns', value=X_train.columns.tolist())
    kwargs['ti'].xcom_push(key='train_shape', value=X_train.shape)
    kwargs['ti'].xcom_push(key='test_shape', value=X_test.shape)
    
    # データをファイルとして保存
    return {
        'X_train_path': os.path.join(TRAIN_DIR, 'X_train.csv'),
        'y_train_path': os.path.join(TRAIN_DIR, 'y_train.csv'),
        'X_test_path': os.path.join(TEST_DIR, 'X_test.csv'),
        'y_test_path': os.path.join(TEST_DIR, 'y_test.csv'),
    }


def train_random_forest(**kwargs):
    """
    ランダムフォレストモデルの学習
    """
    # データパスの取得
    ti = kwargs['ti']
    data_paths = ti.xcom_pull(task_ids='load_training_data')
    
    # データの読み込み
    X_train = pd.read_csv(data_paths['X_train_path'])
    y_train = pd.read_csv(data_paths['y_train_path']).values.ravel()
    X_test = pd.read_csv(data_paths['X_test_path'])
    y_test = pd.read_csv(data_paths['y_test_path']).values.ravel()
    
    # ハイパーパラメータ
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    # MLflowの設定
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # モデルの学習を実行
    with mlflow.start_run(run_name='random_forest') as run:
        # パラメータのログ
        mlflow.log_params(params)
        
        # モデルの初期化と学習
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 評価指標の計算
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # 評価指標のログ
        mlflow.log_metrics(metrics)
        
        # 特徴量重要度のログ
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 特徴量重要度をCSVとして保存してログ
            importance_path = '/tmp/feature_importance_rf.csv'
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
        
        # モデルの保存
        model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
        mlflow.sklearn.save_model(model, model_path)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # 情報をXComに保存
        kwargs['ti'].xcom_push(key='rf_metrics', value=metrics)
        kwargs['ti'].xcom_push(key='rf_run_id', value=run.info.run_id)
        kwargs['ti'].xcom_push(key='rf_model_path', value=model_path)
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'run_id': run.info.run_id
    }


def train_gradient_boosting(**kwargs):
    """
    勾配ブースティングモデルの学習
    """
    # データパスの取得
    ti = kwargs['ti']
    data_paths = ti.xcom_pull(task_ids='load_training_data')
    
    # データの読み込み
    X_train = pd.read_csv(data_paths['X_train_path'])
    y_train = pd.read_csv(data_paths['y_train_path']).values.ravel()
    X_test = pd.read_csv(data_paths['X_test_path'])
    y_test = pd.read_csv(data_paths['y_test_path']).values.ravel()
    
    # ハイパーパラメータ
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    # MLflowの設定
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # モデルの学習を実行
    with mlflow.start_run(run_name='gradient_boosting') as run:
        # パラメータのログ
        mlflow.log_params(params)
        
        # モデルの初期化と学習
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 評価指標の計算
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # 評価指標のログ
        mlflow.log_metrics(metrics)
        
        # 特徴量重要度のログ
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 特徴量重要度をCSVとして保存してログ
            importance_path = '/tmp/feature_importance_gb.csv'
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
        
        # モデルの保存
        model_path = os.path.join(MODEL_DIR, 'gradient_boosting_model.pkl')
        mlflow.sklearn.save_model(model, model_path)
        mlflow.sklearn.log_model(model, "gradient_boosting_model")
        
        # 情報をXComに保存
        kwargs['ti'].xcom_push(key='gb_metrics', value=metrics)
        kwargs['ti'].xcom_push(key='gb_run_id', value=run.info.run_id)
        kwargs['ti'].xcom_push(key='gb_model_path', value=model_path)
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'run_id': run.info.run_id
    }


def select_best_model(**kwargs):
    """
    学習したモデルから最良のものを選択
    """
    ti = kwargs['ti']
    
    # 各モデルの評価指標を取得
    rf_metrics = ti.xcom_pull(task_ids='train_random_forest', key='rf_metrics')
    gb_metrics = ti.xcom_pull(task_ids='train_gradient_boosting', key='gb_metrics')
    
    # F1スコアで比較（必要に応じて別の指標を使用）
    if rf_metrics['f1'] > gb_metrics['f1']:
        best_model = {
            'model_type': 'random_forest',
            'run_id': ti.xcom_pull(task_ids='train_random_forest', key='rf_run_id'),
            'model_path': ti.xcom_pull(task_ids='train_random_forest', key='rf_model_path'),
            'metrics': rf_metrics
        }
    else:
        best_model = {
            'model_type': 'gradient_boosting',
            'run_id': ti.xcom_pull(task_ids='train_gradient_boosting', key='gb_run_id'),
            'model_path': ti.xcom_pull(task_ids='train_gradient_boosting', key='gb_model_path'),
            'metrics': gb_metrics
        }
    
    # 最良モデルをMLflowにベストモデルとしてタグ付け
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_tag(best_model['run_id'], "best_model", "true")
    
    # 最良モデルのパスをファイルに保存（他のパイプラインで使用するため）
    best_model_info_path = os.path.join(MODEL_DIR, 'best_model_info.txt')
    with open(best_model_info_path, 'w') as f:
        f.write(f"model_type={best_model['model_type']}\n")
        f.write(f"run_id={best_model['run_id']}\n")
        f.write(f"model_path={best_model['model_path']}\n")
        f.write(f"accuracy={best_model['metrics']['accuracy']}\n")
        f.write(f"f1_score={best_model['metrics']['f1']}\n")
    
    # 結果をS3にアップロード
    s3_hook = S3Hook(aws_conn_id='aws_default')
    bucket_name = Variable.get('ml_data_bucket', default_var='ml-data-bucket')
    s3_key = f'models/best_model_info_{datetime.now().strftime("%Y%m%d")}.txt'
    
    s3_hook.load_file(
        filename=best_model_info_path,
        bucket_name=bucket_name,
        replace=True,
        key=s3_key
    )
    
    return {
        'best_model_type': best_model['model_type'],
        'best_model_run_id': best_model['run_id'],
        'best_model_path': best_model['model_path'],
        'best_model_accuracy': best_model['metrics']['accuracy'],
        'best_model_f1': best_model['metrics']['f1'],
        's3_info_path': f's3://{bucket_name}/{s3_key}'
    }


# ETLパイプラインが完了するのを待つセンサー
wait_for_etl = ExternalTaskSensor(
    task_id='wait_for_etl',
    external_dag_id='ml_data_etl',
    external_task_id='update_execution_date',
    timeout=3600,
    mode='reschedule',
    allowed_states=['success'],
    failed_states=['failed', 'skipped'],
    dag=dag,
)

# モデルディレクトリの作成
create_model_dir = BashOperator(
    task_id='create_model_dir',
    bash_command=f'mkdir -p {MODEL_DIR}',
    dag=dag,
)

# データ読み込みタスク
load_data = PythonOperator(
    task_id='load_training_data',
    python_callable=load_training_data,
    provide_context=True,
    dag=dag,
)

# モデル学習タスク
train_rf = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    provide_context=True,
    dag=dag,
)

train_gb = PythonOperator(
    task_id='train_gradient_boosting',
    python_callable=train_gradient_boosting,
    provide_context=True,
    dag=dag,
)

# 最良モデル選択タスク
select_model = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    provide_context=True,
    dag=dag,
)

# タスクの依存関係の設定
wait_for_etl >> create_model_dir >> load_data
load_data >> [train_rf, train_gb] >> select_model
