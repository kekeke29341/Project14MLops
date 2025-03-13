"""
データの抽出・変換・ロード (ETL) を行うAirflow DAG
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.models import Variable

import pandas as pd
import numpy as np


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
    'ml_data_etl',
    default_args=default_args,
    description='機械学習モデル用のデータETLパイプライン',
    schedule_interval='0 2 * * *',  # 毎日午前2時に実行
    catchup=False,
)

# 環境変数とパスの設定
DATA_DIR = '/opt/airflow/data'
PROCESSED_DIR = '/opt/airflow/data/processed'
RAW_DIR = '/opt/airflow/data/raw'

# 必要なディレクトリを作成
for directory in [DATA_DIR, PROCESSED_DIR, RAW_DIR]:
    os.makedirs(directory, exist_ok=True)


def extract_data_from_db(**kwargs):
    """
    PostgreSQLからデータを抽出
    """
    pg_hook = PostgresHook(postgres_conn_id='postgres_ml_db')
    sql = """
    SELECT 
        customer_id,
        product_id,
        purchase_date,
        amount,
        discount,
        age,
        gender,
        location,
        category,
        is_returned
    FROM 
        sales_data
    WHERE 
        purchase_date >= CURRENT_DATE - INTERVAL '30 days'
    """
    
    # データをDataFrameとして取得
    df = pg_hook.get_pandas_df(sql)
    
    # ファイルに保存
    out_path = os.path.join(RAW_DIR, 'sales_data.csv')
    df.to_csv(out_path, index=False)
    
    return out_path


def extract_data_from_s3(**kwargs):
    """
    S3からデータを抽出
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    bucket_name = Variable.get('ml_data_bucket', default_var='ml-data-bucket')
    
    # 顧客情報ファイルのダウンロード
    customer_key = 'raw/customer_demographics.csv'
    customer_file = os.path.join(RAW_DIR, 'customer_demographics.csv')
    s3_hook.download_file(
        key=customer_key,
        bucket_name=bucket_name,
        local_path=customer_file
    )
    
    # 製品情報ファイルのダウンロード
    product_key = 'raw/product_features.csv'
    product_file = os.path.join(RAW_DIR, 'product_features.csv')
    s3_hook.download_file(
        key=product_key,
        bucket_name=bucket_name,
        local_path=product_file
    )
    
    return f"Downloaded files: {customer_file}, {product_file}"


def transform_data(**kwargs):
    """
    データの変換と特徴量エンジニアリング
    """
    # 販売データの読み込み
    sales_path = os.path.join(RAW_DIR, 'sales_data.csv')
    df_sales = pd.read_csv(sales_path)
    
    # 顧客データの読み込み
    customer_path = os.path.join(RAW_DIR, 'customer_demographics.csv')
    df_customer = pd.read_csv(customer_path)
    
    # 製品データの読み込み
    product_path = os.path.join(RAW_DIR, 'product_features.csv')
    df_product = pd.read_csv(product_path)
    
    # 日付を日時型に変換
    df_sales['purchase_date'] = pd.to_datetime(df_sales['purchase_date'])
    
    # 曜日と月を特徴量として追加
    df_sales['day_of_week'] = df_sales['purchase_date'].dt.dayofweek
    df_sales['month'] = df_sales['purchase_date'].dt.month
    
    # データの結合
    df_merged = pd.merge(df_sales, df_customer, on='customer_id', how='left')
    df_merged = pd.merge(df_merged, df_product, on='product_id', how='left')
    
    # 欠損値の処理
    df_merged.fillna({
        'age': df_merged['age'].median(),
        'discount': 0,
        'is_returned': 0
    }, inplace=True)
    
    # カテゴリカル変数のエンコーディング
    categorical_columns = ['gender', 'location', 'category']
    for col in categorical_columns:
        dummies = pd.get_dummies(df_merged[col], prefix=col, drop_first=True)
        df_merged = pd.concat([df_merged, dummies], axis=1)
    
    # 不要なカラムの削除
    df_merged.drop(['purchase_date', 'customer_id', 'product_id'] + categorical_columns, 
                  axis=1, inplace=True)
    
    # 顧客セグメントの作成
    df_merged['high_value_customer'] = (df_merged['amount'] > df_merged['amount'].quantile(0.8)).astype(int)
    
    # 特徴量と目標変数の分離
    X = df_merged.drop('is_returned', axis=1)
    y = df_merged['is_returned']
    
    # 処理済みデータの保存
    X_path = os.path.join(PROCESSED_DIR, 'X_features.csv')
    y_path = os.path.join(PROCESSED_DIR, 'y_target.csv')
    
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
    
    return f"Transformed data saved to {X_path} and {y_path}"


def split_train_test(**kwargs):
    """
    データを訓練用とテスト用に分割
    """
    from sklearn.model_selection import train_test_split
    
    # 処理済みデータの読み込み
    X_path = os.path.join(PROCESSED_DIR, 'X_features.csv')
    y_path = os.path.join(PROCESSED_DIR, 'y_target.csv')
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]  # y_targetは1列のみ
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 分割したデータの保存
    train_dir = os.path.join(PROCESSED_DIR, 'train')
    test_dir = os.path.join(PROCESSED_DIR, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(train_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(test_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)
    
    return f"Split data saved to {train_dir} and {test_dir}"


def load_to_s3(**kwargs):
    """
    処理済みデータをS3にアップロード
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    bucket_name = Variable.get('ml_data_bucket', default_var='ml-data-bucket')
    
    # 処理済みデータのディレクトリ
    dirs_to_upload = [PROCESSED_DIR]
    
    # 再帰的にすべてのファイルをアップロード
    for directory in dirs_to_upload:
        for root, _, files in os.walk(directory):
            for file in files:
                local_path = os.path.join(root, file)
                # ローカルパスからS3キーを作成
                s3_key = os.path.join(
                    'processed',
                    os.path.relpath(local_path, PROCESSED_DIR)
                )
                # S3にアップロード
                s3_hook.load_file(
                    filename=local_path,
                    bucket_name=bucket_name,
                    replace=True,
                    key=s3_key
                )
    
    return f"Uploaded processed data to S3 bucket: {bucket_name}"


# タスクの定義
create_data_dirs = BashOperator(
    task_id='create_data_dirs',
    bash_command=f'mkdir -p {DATA_DIR} {PROCESSED_DIR} {RAW_DIR}',
    dag=dag,
)

extract_db_data = PythonOperator(
    task_id='extract_db_data',
    python_callable=extract_data_from_db,
    dag=dag,
)

extract_s3_data = PythonOperator(
    task_id='extract_s3_data',
    python_callable=extract_data_from_s3,
    dag=dag,
)

transform_data_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

split_data_task = PythonOperator(
    task_id='split_train_test',
    python_callable=split_train_test,
    dag=dag,
)

load_s3_task = PythonOperator(
    task_id='load_to_s3',
    python_callable=load_to_s3,
    dag=dag,
)

# データ処理の完了を記録
update_execution_date = PostgresOperator(
    task_id='update_execution_date',
    postgres_conn_id='postgres_ml_db',
    sql="""
    INSERT INTO etl_logs (pipeline_name, execution_date, status)
    VALUES ('ml_data_etl', %s, 'completed')
    """,
    parameters=[datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    dag=dag,
)

# タスクの依存関係を設定
create_data_dirs >> [extract_db_data, extract_s3_data] >> transform_data_task >> split_data_task >> load_s3_task >> update_execution_date
