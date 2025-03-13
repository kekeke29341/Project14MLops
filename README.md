# MLOps Platform

機械学習モデルの開発、学習、デプロイ、モニタリングのためのMLOpsプラットフォーム

## 概要

このリポジトリは、機械学習オペレーション（MLOps）のためのエンドツーエンドの基盤を提供します。モデル開発から本番環境でのデプロイまで、機械学習ライフサイクル全体を管理するためのコンポーネントとツールを含みます。

## システムアーキテクチャ

MLOpsプラットフォームは以下のコンポーネントで構成されています：

```
project_root/
│── models/                # モデル関連のコード・チェックポイント
│   ├── train.py           # Pytorchでのモデル学習スクリプト
│   ├── inference.py       # 推論用スクリプト
│   ├── model.pth          # 学習済みモデルの保存ファイル
│   ├── dataloader.py      # データの前処理・読み込み
│── pipelines/             # MLOps関連のパイプライン
│   ├── airflow/           # AirflowのDAGファイル
│   │   ├── etl_dag.py     # ETLパイプライン
│   │   ├── train_dag.py   # モデル学習のAirflow DAG
│   ├── kubeflow/          # Kubeflow Pipelines用のスクリプト
│   │   ├── train_pipeline.py
│   │   ├── deploy_pipeline.py
│   ├── mlflow/            # MLflowのモデル管理関連
│   │   ├── tracking.py
│   ├── terraform/         # インフラ構築のTerraformスクリプト
│   │   ├── main.tf
│   ├── helm/              # Helmの設定
│   │   ├── Chart.yaml
│── serving/               # モデルデプロイの設定
│   ├── kfserving.yaml     # KFServingの設定ファイル
│   ├── istio.yaml         # Istioの設定
│── monitoring/            # モニタリング関連
│   ├── prometheus.yaml    # Prometheusの設定
│   ├── grafana.yaml       # Grafanaの設定
│── .gitignore             # Git管理対象外の設定
│── README.md              # ドキュメント
```

## 主な機能

- **モデルトレーニング**: PyTorchによる機械学習モデルの開発と学習
- **データパイプライン**: AirflowによるETLプロセスの自動化
- **モデル管理**: MLflowによるモデルのバージョン管理と実験追跡
- **パイプライン自動化**: Kubeflow Pipelinesによるエンドツーエンドのワークフロー
- **モデルデプロイ**: KFServingによるモデルのサービング
- **モニタリング**: PrometheusとGrafanaによるモデルとインフラのモニタリング
- **インフラ管理**: Terraformによるクラウドリソースのプロビジョニングとデプロイ
- **サービスメッシュ**: Istioによるトラフィック管理とセキュリティ

## 前提条件

- Python 3.9以上
- Docker
- Kubernetes 1.21以上
- Helm 3.0以上
- Terraform 1.0以上
- AWS CLIまたはGoogle Cloud SDK（デプロイ先に応じて）

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/mlops-platform.git
cd mlops-platform
```

### 2. 環境のセットアップ

```bash
# Python仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windowsの場合は venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. インフラストラクチャのプロビジョニング

```bash
cd pipelines/terraform
terraform init
terraform plan
terraform apply
```

### 4. Kubernetes環境のセットアップ

```bash
# kubeconfigの設定
aws eks update-kubeconfig --name your-cluster-name --region your-region

# Helmの初期化
helm repo add airflow https://airflow.apache.org/
helm repo add mlflow https://community-charts.github.io/helm-charts
helm repo add prometheus https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo add kfserving https://storage.googleapis.com/kfserving-charts
helm repo update
```

### 5. MLOpsプラットフォームのデプロイ

```bash
cd pipelines/helm
helm install mlops-platform .
```

## モデル学習の実行

```bash
cd models
python train.py --config config.yaml
```

## データパイプラインの実行

Airflow UIを使用してDAGを開始するか、以下のコマンドで直接実行します：

```bash
airflow dags trigger ml_data_etl
```

## モデルのデプロイ

```bash
# Kubeflow Pipelinesを使用したデプロイ
cd pipelines/kubeflow
python deploy_pipeline.py --model-path /path/to/model --model-name your-model-name
```

または、KFServingマニフェストを直接適用：

```bash
kubectl apply -f serving/kfserving.yaml
```

## モニタリング

1. Prometheusダッシュボードへのアクセス：
   - http://prometheus.your-domain.com

2. Grafanaダッシュボードへのアクセス：
   - http://grafana.your-domain.com
   - デフォルト認証情報: admin/admin

## リソース

- [MLflowダッシュボード](http://mlflow.your-domain.com)
- [Airflowダッシュボード](http://airflow.your-domain.com)
- [Kubeflowダッシュボード](http://kubeflow.your-domain.com)

## 貢献

プロジェクトへの貢献をご希望の方は、以下のガイドラインをご確認ください：

1. リポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## ライセンス

[MIT License](LICENSE)
