"""
模型训练脚本
- 三大模型：
  1. XGBoost分类器 - 胜平负
  2. XGBoost回归器 - 主队进球数
  3. XGBoost回归器 - 客队进球数
- 概率校准
- 保存模型
"""
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss
import xgboost as xgb

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.features import prepare_dataset, MatchFeatureBuilder, EloRatingSystem
from model.version_manager import ModelVersionManager

warnings.filterwarnings('ignore')

# 数据路径
DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_DIR = Path(__file__).parent.parent / 'model' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 训练配置
START_YEAR = 2002  # 用2002年后的数据训练（5个世界杯周期）
TEST_SPLIT_DATE = '2023-01-01'  # 时间切分，避免随机切分的泄漏


def load_matches():
    """加载所有比赛数据"""
    df = pd.read_csv(DATA_DIR / 'results.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df['neutral'] = df['neutral'].astype(str).map({'TRUE': True, 'FALSE': False, 'True': True, 'False': False}).fillna(True)
    return df.sort_values('date').reset_index(drop=True)


def prepare_features(matches_df):
    """构建特征矩阵"""
    print("="*70)
    print("特征构建")
    print("="*70)
    features, labels, meta, builder = prepare_dataset(matches_df, start_year=START_YEAR)
    return features, labels, meta, builder


def train_outcome_model(X_train, y_train, X_test, y_test):
    """
    训练胜平负分类器
    label: 1=主胜, 0=平, -1=客胜 -> XGBoost需要0,1,2
    """
    print("\n" + "="*70)
    print("训练胜平负分类器 (XGBoost)")
    print("="*70)

    # 映射标签: -1,0,1 -> 0,1,2 (客胜, 平, 主胜)
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})

    # 类别权重（处理不平衡）
    class_counts = y_train_mapped.value_counts(normalize=True).to_dict()
    total = len(y_train_mapped)
    weights = np.array([total / (3 * class_counts[i]) for i in y_train_mapped])
    sample_weight = weights

    # 训练XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_test, y_test_mapped)],
        sample_weight=sample_weight,
        verbose=False,
    )

    # 评估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test_mapped, y_pred)
    ll = log_loss(y_test_mapped, y_prob)

    print(f"测试准确率: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test_mapped, y_pred,
                                target_names=['客胜', '平局', '主胜']))

    # 特征重要性
    print("\n特征重要性 Top 15:")
    fi = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(15)
    print(fi.to_string(index=False))

    return model, acc, ll


def train_score_model(X_train, y_train, X_test, y_test, target_name='home_score'):
    """
    训练进球数回归器
    使用Poisson目标（适合计数数据）
    """
    print(f"\n训练进球回归器: {target_name} (XGBoost Poisson)")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective='count:poisson',
        eval_metric='poisson-nloglik',
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    y_pred = model.predict(X_test)
    y_pred_round = np.round(y_pred).astype(int)
    y_pred_round = np.clip(y_pred_round, 0, 10)

    # 评估
    mae = np.mean(np.abs(y_pred - y_test))
    exact_acc = np.mean(y_pred_round == y_test)
    within_one = np.mean(np.abs(y_pred_round - y_test) <= 1)

    print(f"  MAE: {mae:.4f}")
    print(f"  精确命中率: {exact_acc:.4f}")
    print(f"  误差<=1: {within_one:.4f}")
    print(f"  预测均值: {y_pred.mean():.3f}  实际均值: {y_test.mean():.3f}")
    return model, mae, exact_acc


def main():
    print("="*70)
    print(f"世界杯预测模型训练 | 训练数据起始年: {START_YEAR}")
    print("="*70)

    # 1. 加载数据
    matches = load_matches()
    print(f"总比赛数据: {len(matches)} 场")

    # 2. 构建特征
    features, labels, meta, builder = prepare_features(matches)

    # 3. 时间切分（避免随机切分导致泄漏）
    test_mask = meta['date'] >= TEST_SPLIT_DATE
    X_train, X_test = features[~test_mask], features[test_mask]
    y_train_result, y_test_result = labels.loc[~test_mask, 'result'], labels.loc[test_mask, 'result']
    y_train_hs, y_test_hs = labels.loc[~test_mask, 'home_score'], labels.loc[test_mask, 'home_score']
    y_train_as, y_test_as = labels.loc[~test_mask, 'away_score'], labels.loc[test_mask, 'away_score']

    print(f"\n训练集: {len(X_train)} 场 ({meta.loc[~test_mask, 'date'].min().date()} ~ {meta.loc[~test_mask, 'date'].max().date()})")
    print(f"测试集: {len(X_test)} 场 ({meta.loc[test_mask, 'date'].min().date()} ~ {meta.loc[test_mask, 'date'].max().date()})")

    # 4. 训练胜平负模型
    outcome_model, outcome_acc, outcome_ll = train_outcome_model(
        X_train, y_train_result, X_test, y_test_result
    )

    # 5. 训练进球数模型
    home_score_model, home_mae, home_acc = train_score_model(
        X_train, y_train_hs, X_test, y_test_hs, 'home_score'
    )
    away_score_model, away_mae, away_acc = train_score_model(
        X_train, y_train_as, X_test, y_test_as, 'away_score'
    )

    # 6. 保存模型和元数据（版本化，不覆盖旧模型）
    print("\n" + "="*70)
    print("保存模型（版本化）...")
    print("="*70)

    # 6a. 重算全历史Elo（在保存前完成，元数据需要 elo_dict）
    print("重算全历史Elo...")
    full_elo = EloRatingSystem()
    for _, m in matches.iterrows():
        full_elo.update(
            m['home_team'], m['away_team'],
            m['home_score'], m['away_score'],
            m['tournament'], m['neutral'], m['date']
        )
    elo_dict = full_elo.to_dict()
    print(f"共 {len(elo_dict)} 支球队的Elo等级分")

    # 6b. 构建训练元数据
    metadata = {
        'start_year': START_YEAR,
        'test_split_date': TEST_SPLIT_DATE,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_count': len(features.columns),
        'outcome_accuracy': float(outcome_acc),
        'outcome_log_loss': float(outcome_ll),
        'home_score_mae': float(home_mae),
        'home_score_exact': float(home_acc),
        'away_score_mae': float(away_mae),
        'away_score_exact': float(away_acc),
        'training_date': pd.Timestamp.now().isoformat(),
        'data_latest_date': matches['date'].max().isoformat(),
        'total_teams': len(elo_dict),
    }

    # 6c. 创建新版本（自动设为活跃）
    vm = ModelVersionManager(MODEL_DIR)
    version_id, version_dir = vm.create_version(metadata, note="")
    print(f"创建模型版本: {version_id}")

    # 6d. 保存所有文件到版本目录
    outcome_model.save_model(version_dir / 'outcome_model.json')
    home_score_model.save_model(version_dir / 'home_score_model.json')
    away_score_model.save_model(version_dir / 'away_score_model.json')

    with open(version_dir / 'feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(list(features.columns), f, ensure_ascii=False, indent=2)

    with open(version_dir / 'elo_ratings.json', 'w', encoding='utf-8') as f:
        json.dump(elo_dict, f, ensure_ascii=False, indent=2)

    with open(version_dir / 'training_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 6e. 同步活跃版本到根目录（向后兼容 predictor 的直接路径加载）
    vm.finalize_version(version_id)

    print(f"\n模型已保存到: {version_dir}")
    print(f"活跃版本: {version_id}")
    all_versions = vm.list_versions()
    print(f"历史版本共 {len(all_versions)} 个")
    print(f"\n[训练结果摘要]")
    for k, v in metadata.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
