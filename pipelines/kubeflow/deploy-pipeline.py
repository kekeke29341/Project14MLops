"""
Kubeflow Pipelinesを使用したモデルデプロイパイプラインの定義
"""

import os
import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath, InputBool, InputInt, InputStr
from typing import NamedTuple, Dict, List
import yaml


# モデル検証コンポーネント
@func_to_container_op
def validate_model(
    model_path: InputPath(),
    validation_results_path: OutputPath(),
    validation_thresholds: str = '{"accuracy": 0.8, "latency_ms": 100}'
) -> NamedTuple('ValidationOutput', [('is_valid', bool), ('validation_msg', str)]):
    """
    デプロイ前にモデルを検証する
    
    Args:
        model_path: 検証するモデルのパス
        validation_results_path: 検証結果の出力パス
        validation_thresholds: 検証閾値のJSON文字列
    """
    import json
    import time
    import pickle
    import numpy as np
    import pandas as pd
    import os
    from collections import namedtuple
    
    # 出力ディレクトリの作成
    os.makedirs(validation_results_path, exist_ok=True)
    
    # 検証閾値の読み込み
    thresholds = json.loads(validation_thresholds)
    
    # モデルメタデータの読み込み
    try:
        with open(os.path.join(model_path, 'package_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        # 通常のモデルメタデータを試す
        with open(os.path.join(model_path, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
    
    # モデルのロード
    with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # 精度の検証
    model_accuracy = metadata.get('metrics', {}).get('accuracy', 0)
    accuracy_valid = model_accuracy >= thresholds.get('accuracy', 0.8)
    
    # レイテンシーの検証
    # ダミーデータで推論の速度をテスト
    num_features = 10  # 特徴量の数は実際のモデルに合わせて調整すべき
    dummy_data = np.random.rand(1000, num_features)
    
    start_time = time.time()
    model.predict(dummy_data)
    end_time = time.time()
    
    # 1000サンプルの推論にかかった時間をミリ秒単位で計算
    batch_latency_ms = (end_time - start_time) * 1000
    avg_latency_ms = batch_latency_ms / 1000  # 1サンプルあたりの平均レイテンシー
    
    latency_valid = avg_latency_ms <= thresholds.get('latency_ms', 100)
    
    # 全体の検証結果
    is_valid = accuracy_valid and latency_valid
    
    # 検証結果のまとめ
    validation_results = {
        'accuracy': {
            'value': model_accuracy,
            'threshold': thresholds.get('accuracy', 0.8),
            'is_valid': accuracy_valid
        },
        'latency_ms': {
            'value': avg_latency_ms,
            'threshold': thresholds.get('latency_ms', 100),
            'is_valid': latency_valid
        },
        'is_valid': is_valid,
        'model_metadata': metadata
    }
    
    # 結果の保存
    with open(os.path.join(validation_results_path, 'validation_results.json'), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # メッセージの生成
    if is_valid:
        validation_msg = "モデル検証に成功しました。デプロイを続行します。"
    else:
        reasons = []
        if not accuracy_valid:
            reasons.append(f"精度が閾値を下回っています ({model_accuracy:.4f} < {thresholds.get('accuracy', 0.8):.4f})")
        if not latency_valid:
            reasons.append(f"レイテンシーが閾値を超えています ({avg_latency_ms:.2f}ms > {thresholds.get('latency_ms', 100)}ms)")
        
        validation_msg = "モデル検証に失敗しました: " + "; ".join(reasons)
    
    # 出力の生成
    ValidationOutput = namedtuple('ValidationOutput', ['is_valid', 'validation_msg'])
    return ValidationOutput(is_valid, validation_msg)


# モデルのONNX形式へのエクスポート
@func_to_container_op
def export_to_onnx(
    model_path: InputPath(),
    onnx_model_path: OutputPath()
) -> str:
    """
    モデルをONNX形式にエクスポート
    
    Args:
        model_path: 学習済みモデルのパス
        onnx_model_path: ONNXモデルの出力パス
    """
    import pickle
    import json
    import numpy as np
    import os
    
    # ONNXのインポート
    try:
        import onnx
        import skl2onnx
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        os.system('pip install onnx skl2onnx')
        import onnx
        import skl2onnx
        from skl2onnx.common.data_types import FloatTensorType
    
    # 出力ディレクトリの作成
    os.makedirs(onnx_model_path, exist_ok=True)
    
    # モデルのロード
    with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # メタデータの読み込み
    try:
        with open(os.path.join(model_path, 'package_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        with open(os.path.join(model_path, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
    
    # 特徴量の数を取得（メタデータから取得するか、推定する）
    if 'num_features' in metadata:
        num_features = metadata['num_features']
    else:
        # モデルのタイプによって特徴量の数を推定
        model_type = metadata.get('model_type', '')
        if model_type == 'random_forest':
            if hasattr(model, 'n_features_in_'):
                num_features = model.n_features_in_
            elif hasattr(model, 'n_features_'):
                num_features = model.n_features_
            else:
                # デフォルト値
                num_features = 10
        else:
            num_features = 10
    
    # 入力特徴量の型定義
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    # ONNXモデルに変換
    try:
        onnx_model = skl2onnx.convert_sklearn(
            model, 
            initial_types=initial_type,
            target_opset=12
        )
        
        # ONNXモデルの保存
        onnx_file = os.path.join(onnx_model_path, 'model.onnx')
        with open(onnx_file, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        # メタデータをコピー
        with open(os.path.join(onnx_model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return f"モデルをONNX形式にエクスポートしました: {onnx_file}"
    
    except Exception as e:
        # 変換に失敗した場合はエラーメッセージを返す
        error_msg = f"ONNXエクスポートに失敗しました: {str(e)}"
        with open(os.path.join(onnx_model_path, 'error.txt'), 'w') as f:
            f.write(error_msg)
        
        # 元のモデルをそのままコピー
        with open(os.path.join(model_path, 'model.pkl'), 'rb') as src:
            with open(os.path.join(onnx_model_path, 'model.pkl'), 'wb') as dst:
                dst.write(src.read())
        
        # メタデータをコピー
        with open(os.path.join(onnx_model_path, 'metadata.json'), 'w') as f:
            json.dump({**metadata, 'onnx_conversion_failed': True}, f, indent=2)
        
        return error_msg


# KFServingマニフェスト生成コンポーネント
@func_to_container_op
def generate_kfserving_manifest(
    model_path: InputPath(),
    manifest_path: OutputPath(),
    model_name: str,
    namespace: str = 'kubeflow',
    replicas: int = 2,
    framework: str = 'sklearn',
    service_account: str = 'kfserving-sa'
) -> str:
    """
    KFServingマニフェストを生成する
    
    Args:
        model_path: モデルのパス
        manifest_path: 生成されたマニフェストの出力パス
        model_name: モデル名
        namespace: Kubernetesネームスペース
        replicas: レプリカ数
        framework: モデルフレームワーク
        service_account: サービスアカウント名
    """
    import os
    import json
    import yaml
    from datetime import datetime
    
    # 出力ディレクトリの作成
    os.makedirs(manifest_path, exist_ok=True)
    
    # フレームワークの検証
    valid_frameworks = ['sklearn', 'xgboost', 'tensorflow', 'onnx']
    if framework not in valid_frameworks:
        framework = 'sklearn'  # デフォルト値
    
    # メタデータの読み込み
    try:
        metadata_file = os.path.join(model_path, 'metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        try:
            metadata_file = os.path.join(model_path, 'package_metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata_file = os.path.join(model_path, 'model_metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    
    # バージョンの取得
    version = metadata.get('version', '1.0.0')
    
    # リソース設定
    resources = {
        'requests': {
            'cpu': '100m',
            'memory': '512Mi'
        },
        'limits': {
            'cpu': '1',
            'memory': '1Gi'
        }
    }
    
    # KFServingマニフェストの作成
    kfserving_manifest = {
        'apiVersion': 'serving.kubeflow.org/v1beta1',
        'kind': 'InferenceService',
        'metadata': {
            'name': model_name,
            'namespace': namespace,
            'annotations': {
                'serving.kubeflow.org/deploymentMode': 'Serverless',
                'serving.kubeflow.org/autoscalerClass': 'hpa',
                'serving.kubeflow.org/autoscalerMetric': 'concurrency',
                'serving.kubeflow.org/autoscalerTarget': '1',
                'serving.kubeflow.org/minReplicas': '1',
                'serving.kubeflow.org/maxReplicas': str(replicas)
            }
        },
        'spec': {
            'predictor': {
                'serviceAccountName': service_account,
                'minReplicas': 1,
                'maxReplicas': replicas,
                framework.lower(): {
                    'storageUri': 'pvc://model-pvc/models/' + model_name,
                    'resources': resources
                }
            }
        }
    }
    
    # ONNXの場合は特別な設定
    if framework.lower() == 'onnx':
        kfserving_manifest['spec']['predictor']['onnx'] = {
            'storageUri': 'pvc://model-pvc/models/' + model_name,
            'resources': resources
        }
    
    # マニフェストの保存
    manifest_file = os.path.join(manifest_path, 'kfserving.yaml')
    with open(manifest_file, 'w') as f:
        yaml.dump(kfserving_manifest, f, default_flow_style=False)
    
    # モデルデプロイスクリプトの生成
    deploy_script = f"""#!/bin/bash
# モデルデプロイスクリプト
# 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# 前提条件:
# - kubectl がインストールされていること
# - kubeflow がデプロイされていること
# - 適切な権限を持っていること

# モデルをデプロイ
kubectl apply -f kfserving.yaml

echo "モデル {model_name} のデプロイを開始しました"
echo "ステータスを確認するには次のコマンドを実行してください:"
echo "kubectl get inferenceservices -n {namespace} {model_name}"
"""
    
    # デプロイスクリプトの保存
    script_file = os.path.join(manifest_path, 'deploy.sh')
    with open(script_file, 'w') as f:
        f.write(deploy_script)
    
    # スクリプトに実行権限を付与
    os.chmod(script_file, 0o755)
    
    return f"KFServingマニフェストを生成しました: {manifest_file}"


# モデルデプロイコンポーネント
@func_to_container_op
def deploy_model(
    manifest_path: InputPath(),
    model_path: InputPath(),
    deployment_result_path: OutputPath(),
    model_name: str,
    namespace: str = 'kubeflow',
    deploy: bool = False
) -> NamedTuple('DeployOutput', [('deployed', bool), ('service_url', str)]):
    """
    モデルをKFServingにデプロイする
    
    Args:
        manifest_path: KFServingマニフェストのパス
        model_path: モデルのパス
        deployment_result_path: デプロイ結果の出力パス
        model_name: モデル名
        namespace: Kubernetesネームスペース
        deploy: 実際にデプロイするかどうか
    """
    import os
    import json
    import yaml
    import subprocess
    import time
    from collections import namedtuple
    
    # 出力ディレクトリの作成
    os.makedirs(deployment_result_path, exist_ok=True)
    
    # マニフェストの読み込み
    manifest_file = os.path.join(manifest_path, 'kfserving.yaml')
    with open(manifest_file, 'r') as f:
        manifest = yaml.safe_load(f)
    
    deployment_status = {
        'model_name': model_name,
        'namespace': namespace,
        'deployed': False,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'manifest_path': manifest_file,
        'errors': []
    }
    
    # 実際のデプロイを行うかどうか
    if not deploy:
        deployment_status['message'] = 'デプロイはスキップされました（dry run）'
        
        # 結果を保存
        with open(os.path.join(deployment_result_path, 'deployment_result.json'), 'w') as f:
            json.dump(deployment_status, f, indent=2)
        
        DeployOutput = namedtuple('DeployOutput', ['deployed', 'service_url'])
        return DeployOutput(False, '')
    
    try:
        # モデルを PVC にコピー（実際の環境では kubectl cp や別の方法が必要かもしれない）
        os.system(f'mkdir -p /mnt/models/{model_name}')
        os.system(f'cp -r {model_path}/* /mnt/models/{model_name}/')
        
        # マニフェストの適用
        with open('/tmp/kfserving.yaml', 'w') as f:
            yaml.dump(manifest, f)
        
        result = subprocess.run(
            ['kubectl', 'apply', '-f', '/tmp/kfserving.yaml'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            deployment_status['errors'].append(result.stderr)
            raise Exception(f"デプロイに失敗しました: {result.stderr}")
        
        # サービスURLの取得（実際の環境に合わせて調整が必要）
        service_url = f"http://{model_name}.{namespace}.svc.cluster.local/v1/models/{model_name}:predict"
        
        deployment_status.update({
            'deployed': True,
            'service_url': service_url,
            'message': 'デプロイが成功しました',
            'kubectl_output': result.stdout
        })
    
    except Exception as e:
        deployment_status.update({
            'deployed': False,
            'message': f'デプロイに失敗しました: {str(e)}',
            'errors': deployment_status.get('errors', []) + [str(e)]
        })
    
    # 結果を保存
    with open(os.path.join(deployment_result_path, 'deployment_result.json'), 'w') as f:
        json.dump(deployment_status, f, indent=2)
    
    DeployOutput = namedtuple('DeployOutput', ['deployed', 'service_url'])
    return DeployOutput(deployment_status['deployed'], deployment_status.get('service_url', ''))


# KubeflowパイプラインのDAG定義
@dsl.pipeline(
    name='Model Deployment Pipeline',
    description='ML models deployment pipeline with KFServing'
)
def deployment_pipeline(
    model_path: str = 'gs://ml-models/model',
    model_name: str = 'customer-churn-predictor',
    namespace: str = 'kubeflow',
    replicas: int = 2,
    validation_thresholds: str = '{"accuracy": 0.8, "latency_ms": 100}',
    deploy: bool = False
):
    """
    モデルデプロイパイプラインのメイン定義
    
    Args:
        model_path: モデルのGCSパス
        model_name: デプロイするモデルの名前
        namespace: Kubernetesネームスペース
        replicas: レプリカ数
        validation_thresholds: 検証閾値のJSON文字列
        deploy: 実際にデプロイするかどうか
    """
    # ボリュームの設定
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="pipeline-pvc",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWO
    )
    
    # モデルのダウンロード
    download = dsl.ContainerOp(
        name='download_model',
        image='google/cloud-sdk:slim',
        command=['sh', '-c'],
        arguments=[f'mkdir -p /data/model && gsutil -m cp -r {model_path}/* /data/model/'],
        pvolumes={"/data": vop.volume}
    )
    
    # モデルの検証
    validate = validate_model(
        model_path='/data/model',
        validation_results_path='/data/validation',
        validation_thresholds=validation_thresholds
    ).after(download).set_pvolumes({"/data": vop.volume})
    
    # モデルのONNXエクスポート（検証成功時のみ）
    with dsl.Condition(validate.outputs['is_valid'] == 'true'):
        export_onnx = export_to_onnx(
            model_path='/data/model',
            onnx_model_path='/data/onnx'
        ).set_pvolumes({"/data": vop.volume})
        
        # KFServingマニフェストの生成
        generate_manifest = generate_kfserving_manifest(
            model_path='/data/onnx',
            manifest_path='/data/manifest',
            model_name=model_name,
            namespace=namespace,
            replicas=replicas,
            framework='onnx',
            service_account='kfserving-sa'
        ).after(export_onnx).set_pvolumes({"/data": vop.volume})
        
        # モデルのデプロイ
        deploy_op = deploy_model(
            manifest_path='/data/manifest',
            model_path='/data/onnx',
            deployment_result_path='/data/deployment',
            model_name=model_name,
            namespace=namespace,
            deploy=deploy
        ).after(generate_manifest).set_pvolumes({"/data": vop.volume})
    
    # リソース制約の設定
    for op in [validate, generate_manifest]:
        op.set_memory_request('1G')
        op.set_memory_limit('2G')
        op.set_cpu_request('0.5')
        op.set_cpu_limit('1')


# パイプラインのコンパイル
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=deployment_pipeline,
        package_path='deployment_pipeline.yaml'
    )
