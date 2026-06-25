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
from sklearn.model_selection import train_test_split
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
START_YEAR = 2002  # 用2002年后的数据训练（包含2026 WC，学习当前赛事模式）
# 测试切分策略：
#   训练集 = 全部数据（含2026 WC）随机80%
#   验证集 = 随机20%（用于早停和评估）
#   不再使用时间切分——2026数据是最新的赛事模式，必须参与训练
TEST_SPLIT_DATE = '2027-01-01'  # 远在数据之后，确保全量参与训练
VAL_RATIO = 0.2  # 随机验证集比例
RANDOM_SEED = 42


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


def train_outcome_model(X_train, y_train, X_val, y_val):
    """
    训练胜平负分类器
    label: 1=主胜, 0=平, -1=客胜 -> XGBoost需要0,1,2
    """
    print("\n" + "="*70)
    print("训练胜平负分类器 (XGBoost)")
    print("="*70)

    # 映射标签: -1,0,1 -> 0,1,2 (客胜, 平, 主胜)
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

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
        eval_set=[(X_val, y_val_mapped)],
        sample_weight=sample_weight,
        verbose=False,
    )

    # 评估
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    acc = accuracy_score(y_val_mapped, y_pred)
    ll = log_loss(y_val_mapped, y_prob)

    print(f"验证准确率: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val_mapped, y_pred,
                                target_names=['客胜', '平局', '主胜']))

    # 特征重要性
    print("\n特征重要性 Top 15:")
    fi = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(15)
    print(fi.to_string(index=False))

    return model, acc, ll


def train_score_model(X_train, y_train, X_val, y_val, target_name='home_score'):
    """
    训练进球数回归器（保守模式）
    使用Poisson目标（适合计数数据，方差=均值）
    """
    print(f"\n训练进球回归器: {target_name} (XGBoost Poisson) [保守模式]")
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
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(X_val)
    y_pred_round = np.round(y_pred).astype(int)
    y_pred_round = np.clip(y_pred_round, 0, 10)

    # 评估
    mae = np.mean(np.abs(y_pred - y_val))
    exact_acc = np.mean(y_pred_round == y_val)
    within_one = np.mean(np.abs(y_pred_round - y_val) <= 1)

    print(f"  MAE: {mae:.4f}")
    print(f"  精确命中率: {exact_acc:.4f}")
    print(f"  误差<=1: {within_one:.4f}")
    print(f"  预测均值: {y_pred.mean():.3f}  实际均值: {y_val.mean():.3f}")
    return model, mae, exact_acc


def train_aggressive_outcome_model(X_train, y_train, X_val, y_val):
    """
    训练胜平负分类器（激进模式）
    - 更深树、更弱正则化，敢于捕捉极端模式
    """
    print("\n" + "="*70)
    print("训练胜平负分类器 (XGBoost) [激进模式]")
    print("="*70)

    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

    class_counts = y_train_mapped.value_counts(normalize=True).to_dict()
    total = len(y_train_mapped)
    weights = np.array([total / (3 * class_counts[i]) for i in y_train_mapped])

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,              # ↑ 更深（保守:5）
        learning_rate=0.06,       # ↓ 略低学习率配合更多树
        subsample=0.9,            # ↑ 更多样本（保守:0.85）
        colsample_bytree=0.95,    # ↑ 更多特征（保守:0.85）
        min_child_weight=1,       # ↓ 允许小叶子（保守:3）
        gamma=0.0,                # ↓ 无分裂门槛（保守:0.1）
        reg_alpha=0.0,            # ↓ 无L1正则（保守:0.1）
        reg_lambda=0.5,           # ↓ 弱L2正则（保守:1.0）
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_val, y_val_mapped)],
        sample_weight=weights,
        verbose=False,
    )

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    acc = accuracy_score(y_val_mapped, y_pred)
    ll = log_loss(y_val_mapped, y_prob)

    print(f"测试准确率: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val_mapped, y_pred,
                                target_names=['客胜', '平局', '主胜']))

    print("\n特征重要性 Top 15:")
    fi = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(15)
    print(fi.to_string(index=False))

    return model, acc, ll


def train_aggressive_score_model(X_train, y_train, X_val, y_val, target_name='home_score'):
    """
    训练进球数回归器（激进模式）
    - reg:tweedie 替代 count:poisson，允许方差>均值（过度离散）
    - 更深树、更弱正则化
    - 特征含赛事走势，能感知当前杯赛的高进球趋势
    """
    print(f"\n训练进球回归器: {target_name} (XGBoost Tweedie) [激进模式]")
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,              # ↑ 更深（保守:4）
        learning_rate=0.06,       # ↓ 略低学习率配合更多树
        subsample=0.9,            # ↑ 更多样本（保守:0.85）
        colsample_bytree=0.95,    # ↑ 更多特征（保守:0.85）
        min_child_weight=1,       # ↓ 允许小叶子（保守:5）
        gamma=0.0,                # ↓ 无分裂门槛（保守:0.2）
        reg_alpha=0.0,            # ↓ 无L1正则（保守:0.2）
        reg_lambda=0.5,           # ↓ 弱L2正则（保守:1.0）
        random_state=42,
        n_jobs=-1,
        objective='reg:tweedie',  # ← 核心改变：允许过度离散
        tweedie_variance_power=1.5,  # 1.0=Poisson, 2.0=Gamma, 1.5=中间态
        eval_metric='tweedie-nloglik@1.5',
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(X_val)
    y_pred_round = np.round(y_pred).astype(int)
    y_pred_round = np.clip(y_pred_round, 0, 10)

    mae = np.mean(np.abs(y_pred - y_val))
    exact_acc = np.mean(y_pred_round == y_val)
    within_one = np.mean(np.abs(y_pred_round - y_val) <= 1)

    print(f"  MAE: {mae:.4f}")
    print(f"  精确命中率: {exact_acc:.4f}")
    print(f"  误差<=1: {within_one:.4f}")
    print(f"  预测均值: {y_pred.mean():.3f}  实际均值: {y_val.mean():.3f}")
    print(f"  预测标准差: {y_pred.std():.3f}  实际标准差: {y_val.std():.3f}")
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

    # 3. 数据切分：随机 80/20（全量数据含 2026 WC 参与训练）
    # 不再使用时间切分——2026 数据代表当前赛事模式，必须让模型学习
    # 特征构建过程已保证严格时间顺序（无未来信息泄漏）
    X_train, X_val, y_train_result, y_val_result = train_test_split(
        features, labels['result'], test_size=VAL_RATIO, random_state=RANDOM_SEED
    )
    # 为进球数标签保持与胜平负相同的数据划分
    # 用第一次切分的索引保持一致
    _, _, y_train_hs, y_val_hs = train_test_split(
        features, labels['home_score'], test_size=VAL_RATIO, random_state=RANDOM_SEED
    )
    _, _, y_train_as, y_val_as = train_test_split(
        features, labels['away_score'], test_size=VAL_RATIO, random_state=RANDOM_SEED
    )

    print(f"\n训练集: {len(X_train)} 场 (含 2002-2026 全量数据)")
    print(f"验证集: {len(X_val)} 场 (随机 {VAL_RATIO*100:.0f}%)")
    print(f"  日期范围: {meta['date'].min().date()} ~ {meta['date'].max().date()}")
    
    # 统计训练集中 2026 WC 占比
    train_dates_2026 = (meta.loc[X_train.index, 'date'].dt.year == 2026).sum()
    val_dates_2026 = (meta.loc[X_val.index, 'date'].dt.year == 2026).sum()
    print(f"  其中 2026 年比赛: 训练 {train_dates_2026} 场 + 验证 {val_dates_2026} 场 = {train_dates_2026 + val_dates_2026} 场")

    # 4. 训练胜平负模型
    outcome_model, outcome_acc, outcome_ll = train_outcome_model(
        X_train, y_train_result, X_val, y_val_result
    )

    # 5. 训练进球数模型（保守）
    home_score_model, home_mae, home_acc = train_score_model(
        X_train, y_train_hs, X_val, y_val_hs, 'home_score'
    )
    away_score_model, away_mae, away_acc = train_score_model(
        X_train, y_train_as, X_val, y_val_as, 'away_score'
    )

    # 5b. 训练激进模型（新特征 + Tweedie + 放松正则化）
    agg_outcome_model, agg_outcome_acc, agg_outcome_ll = train_aggressive_outcome_model(
        X_train, y_train_result, X_val, y_val_result
    )
    agg_home_score_model, agg_home_mae, agg_home_acc = train_aggressive_score_model(
        X_train, y_train_hs, X_val, y_val_hs, 'home_score'
    )
    agg_away_score_model, agg_away_mae, agg_away_acc = train_aggressive_score_model(
        X_train, y_train_as, X_val, y_val_as, 'away_score'
    )

    # 激进 vs 保守 对比报告
    print("\n" + "="*70)
    print("激进 vs 保守 对比报告")
    print("="*70)
    print(f"{'指标':<20} {'保守':>10} {'激进':>10} {'变化':>10}")
    print("-"*50)
    print(f"{'胜平负准确率':<20} {outcome_acc:>10.4f} {agg_outcome_acc:>10.4f} {'-' if agg_outcome_acc < outcome_acc else '↑':>10}")
    print(f"{'主队进球MAE':<20} {home_mae:>10.4f} {agg_home_mae:>10.4f} {'↓(更好)' if agg_home_mae < home_mae else '↑':>10}")
    print(f"{'客队进球MAE':<20} {away_mae:>10.4f} {agg_away_mae:>10.4f} {'↓(更好)' if agg_away_mae < away_mae else '↑':>10}")
    print(f"{'主队进球命中':<20} {home_acc:>10.4f} {agg_home_acc:>10.4f} {'↑' if agg_home_acc > home_acc else '↓':>10}")
    print(f"{'客队进球命中':<20} {away_acc:>10.4f} {agg_away_acc:>10.4f} {'↑' if agg_away_acc > away_acc else '↓':>10}")

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
        'split_method': 'random_80_20',
        'data_includes_2026_wc': True,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'feature_count': len(features.columns),
        # 保守模型
        'outcome_accuracy': float(outcome_acc),
        'outcome_log_loss': float(outcome_ll),
        'home_score_mae': float(home_mae),
        'home_score_exact': float(home_acc),
        'away_score_mae': float(away_mae),
        'away_score_exact': float(away_acc),
        # 激进模型
        'agg_outcome_accuracy': float(agg_outcome_acc),
        'agg_outcome_log_loss': float(agg_outcome_ll),
        'agg_home_score_mae': float(agg_home_mae),
        'agg_home_score_exact': float(agg_home_acc),
        'agg_away_score_mae': float(agg_away_mae),
        'agg_away_score_exact': float(agg_away_acc),
        # 版本信息
        'model_mode': 'dual',  # conservative + aggressive
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
    agg_outcome_model.save_model(version_dir / 'aggressive_outcome_model.json')
    agg_home_score_model.save_model(version_dir / 'aggressive_home_score_model.json')
    agg_away_score_model.save_model(version_dir / 'aggressive_away_score_model.json')

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
