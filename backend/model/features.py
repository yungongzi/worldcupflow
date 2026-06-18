"""
特征工程模块
所有特征严格按时间顺序计算，避免数据泄漏
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List


# 赛事重要性权重（用于Elo加权和特征）
TOURNAMENT_WEIGHT = {
    'FIFA World Cup': 60,
    'FIFA World Cup qualification': 40,
    'UEFA Euro': 50,
    'Copa América': 50,
    'African Cup of Nations': 50,
    'AFC Asian Cup': 50,
    'Gold Cup': 40,
    'UEFA Euro qualification': 30,
    'African Cup of Nations qualification': 30,
    'AFC Asian Cup qualification': 30,
    'CONCACAF Nations League': 25,
    'UEFA Nations League': 35,
    'Confederations Cup': 45,
    'Friendly': 20,
}

# 默认权重
DEFAULT_WEIGHT = 25


def get_tournament_weight(tournament: str) -> int:
    """获取赛事权重"""
    if tournament in TOURNAMENT_WEIGHT:
        return TOURNAMENT_WEIGHT[tournament]
    # 通用匹配
    t = tournament.lower()
    if 'world cup' in t and 'qualif' not in t:
        return 60
    if 'qualif' in t or 'qualification' in t:
        return 30
    if 'cup' in t or 'championship' in t:
        return 35
    return DEFAULT_WEIGHT


class EloRatingSystem:
    """
    Elo等级分系统
    - 支持赛事权重
    - 支持主场优势
    - 支持时间衰减（可选）
    - 支持动态更新
    """
    def __init__(self, k_base=20, home_advantage=65, initial=1500):
        self.ratings = defaultdict(lambda: initial)
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.initial = initial
        self.history = []  # (date, team, rating)

    def expected_score(self, rating_a, rating_b, neutral=True, a_is_home=False):
        """计算A队预期得分"""
        if neutral:
            adj = 0
        else:
            adj = self.home_advantage if a_is_home else -self.home_advantage
        return 1 / (1 + 10 ** ((rating_b - rating_a - adj) / 400))

    def update(self, home_team, away_team, home_score, away_score,
               tournament='Friendly', neutral=False, date=None):
        """更新两队Elo"""
        # K值随赛事重要性调整
        weight = get_tournament_weight(tournament)
        k = self.k_base * (weight / 30)  # 标准化到30=基准

        rh, ra = self.ratings[home_team], self.ratings[away_team]

        # 实际得分
        if home_score > away_score:
            actual_h = 1.0
        elif home_score == away_score:
            actual_h = 0.5
        else:
            actual_h = 0.0

        # 预期得分（考虑主场）
        expected_h = self.expected_score(rh, ra, neutral=neutral, a_is_home=True)

        # 进球差调节（高比分差时增大K）
        goal_diff = abs(home_score - away_score)
        if goal_diff == 2:
            k *= 1.5
        elif goal_diff == 3:
            k *= 1.75
        elif goal_diff >= 4:
            k *= 1.75 + (goal_diff - 3) * 0.5

        delta = k * (actual_h - expected_h)
        self.ratings[home_team] = rh + delta
        self.ratings[away_team] = ra - delta

        if date is not None:
            self.history.append((date, home_team, self.ratings[home_team]))
            self.history.append((date, away_team, self.ratings[away_team]))

        return self.ratings[home_team], self.ratings[away_team]

    def get_rating(self, team):
        return self.ratings[team]

    def to_dict(self):
        return dict(self.ratings)


class MatchFeatureBuilder:
    """
    比赛特征构建器
    严格按时间顺序遍历所有比赛，为每场比赛构建只用历史数据的特征
    """
    def __init__(self):
        self.elo = EloRatingSystem()
        # 球队近期表现：dict[team] = list of (date, goals_for, goals_against, result, tournament_weight, opponent)
        self.team_history = defaultdict(list)
        # 对战记录：dict[frozenset({team_a, team_b})] = list of (date, result_from_home_perspective, home_team)
        self.h2h_history = defaultdict(list)
        # 球队最近一次比赛日期
        self.last_match_date = {}

    def _recent_form(self, team, current_date, n=10):
        """获取球队近n场比赛"""
        # 过滤掉当前日期之后的（避免泄漏）
        past = [h for h in self.team_history[team] if h[0] < current_date]
        return past[-n:] if past else []

    def _compute_form_features(self, team, current_date, n=10):
        """计算球队近期状态特征"""
        recent = self._recent_form(team, current_date, n)
        if not recent:
            return {
                f'form_{n}_winrate': 0.5,
                f'form_{n}_goaldiff_avg': 0.0,
                f'form_{n}_goals_scored_avg': 1.2,
                f'form_{n}_goals_conceded_avg': 1.2,
                f'form_{n}_weighted_points': 0.5,  # 赛事加权积分
            }
        winrates = []
        goaldiffs = []
        scoreds = []
        conceded = []
        weighted_points = []
        for h in recent:
            gf, ga, result, weight, opp = h[1], h[2], h[3], h[4], h[5]
            winrates.append(1 if result == 1 else (0.5 if result == 0 else 0))
            goaldiffs.append(gf - ga)
            scoreds.append(gf)
            conceded.append(ga)
            # 加权：1=胜=3分，0=平=1分，-1=负=0分，乘以权重
            pts = 3 if result == 1 else (1 if result == 0 else 0)
            weighted_points.append(pts * weight / 30)
        return {
            f'form_{n}_winrate': np.mean(winrates),
            f'form_{n}_goaldiff_avg': np.mean(goaldiffs),
            f'form_{n}_goals_scored_avg': np.mean(scoreds),
            f'form_{n}_goals_conceded_avg': np.mean(conceded),
            f'form_{n}_weighted_points': np.mean(weighted_points),
        }

    def _compute_h2h_features(self, home_team, away_team, current_date, n=5):
        """对战记录特征"""
        key = frozenset({home_team, away_team})
        past = [h for h in self.h2h_history[key] if h[0] < current_date]
        recent = past[-n:] if past else []
        if not recent:
            return {
                'h2h_home_winrate': 0.5,
                'h2h_draw_rate': 0.33,
                'h2h_count': 0,
                'h2h_home_goal_diff_avg': 0.0,
            }
        home_wins = 0
        draws = 0
        # 从home_team视角计算
        home_gd_list = []
        for h in recent:
            date, result, home_in_match = h[0], h[1], h[2]
            # result是1/0/-1，是从home_in_match视角
            # 转换到当前home_team视角
            if home_in_match == home_team:
                # 当前home_team就是当时的主队
                cur_persp = result
                gd = h[3]  # 当时主队进球差
            else:
                # 当前home_team是当时的客队，反转
                cur_persp = -result
                gd = -h[3]
            if cur_persp == 1:
                home_wins += 1
            elif cur_persp == 0:
                draws += 1
            home_gd_list.append(gd)
        return {
            'h2h_home_winrate': home_wins / len(recent),
            'h2h_draw_rate': draws / len(recent),
            'h2h_count': len(past),
            'h2h_home_goal_diff_avg': np.mean(home_gd_list) if home_gd_list else 0,
        }

    def build_features_for_match(self, home_team, away_team, date, tournament, neutral):
        """为单场比赛构建特征（基于截至当前的历史）"""
        # Elo特征
        elo_h = self.elo.get_rating(home_team)
        elo_a = self.elo.get_rating(away_team)
        elo_diff = elo_h - elo_a
        elo_ratio = elo_h / max(elo_a, 1)

        # 近期状态（10场）
        form_h = self._compute_form_features(home_team, date, n=10)
        form_a = self._compute_form_features(away_team, date, n=10)

        # 近期状态（5场 - 更近期）
        form_h_5 = self._compute_form_features(home_team, date, n=5)
        form_a_5 = self._compute_form_features(away_team, date, n=5)

        # 对战记录
        h2h = self._compute_h2h_features(home_team, away_team, date, n=5)

        # 赛事权重
        tour_weight = get_tournament_weight(tournament)

        # 上场比赛间隔（天）
        days_since_h = (date - self.last_match_date[home_team]).days if home_team in self.last_match_date else 90
        days_since_a = (date - self.last_match_date[away_team]).days if away_team in self.last_match_date else 90
        # 截断到合理范围
        days_since_h = min(max(days_since_h, 3), 365)
        days_since_a = min(max(days_since_a, 3), 365)

        # 综合特征
        features = {
            'elo_home': elo_h,
            'elo_away': elo_a,
            'elo_diff': elo_diff,
            'elo_ratio': elo_ratio,
            'elo_diff_normalized': elo_diff / 400,  # 标准化
            'neutral': int(neutral),
            'tournament_weight': tour_weight,
            'days_since_home_last': days_since_h,
            'days_since_away_last': days_since_a,
            'days_diff': days_since_h - days_since_a,
        }
        # 主队近10场特征
        for k, v in form_h.items():
            features[f'home_{k}'] = v
        # 客队近10场特征
        for k, v in form_a.items():
            features[f'away_{k}'] = v
        # 主客队近5场特征
        for k, v in form_h_5.items():
            features[f'home_{k}'] = v
        for k, v in form_a_5.items():
            features[f'away_{k}'] = v
        # 对战记录
        features.update(h2h)
        # 衍生特征
        features['form_diff_winrate'] = form_h['form_10_winrate'] - form_a['form_10_winrate']
        features['form_diff_goaldiff'] = form_h['form_10_goaldiff_avg'] - form_a['form_10_goaldiff_avg']
        features['form_diff_5_winrate'] = form_h_5['form_5_winrate'] - form_a_5['form_5_winrate']
        features['attack_vs_defense_home'] = form_h['form_10_goals_scored_avg'] * form_a['form_10_goals_conceded_avg']
        features['attack_vs_defense_away'] = form_a['form_10_goals_scored_avg'] * form_h['form_10_goals_conceded_avg']

        return features

    def update_with_match(self, home_team, away_team, home_score, away_score,
                          date, tournament='Friendly', neutral=False):
        """用一场已结束的比赛更新内部状态"""
        result = 1 if home_score > away_score else (0 if home_score == away_score else -1)
        weight = get_tournament_weight(tournament)

        # 更新Elo
        self.elo.update(home_team, away_team, home_score, away_score,
                        tournament=tournament, neutral=neutral, date=date)

        # 更新球队历史
        # 主队视角：(date, goals_for, goals_against, result, weight, opponent)
        self.team_history[home_team].append((date, home_score, away_score, result, weight, away_team))
        # 客队视角：result反转
        self.team_history[away_team].append((date, away_score, home_score, -result, weight, home_team))

        # 更新对战记录：key是frozenset，记录当时的result（从主队视角）和当时的主队
        key = frozenset({home_team, away_team})
        gd = home_score - away_score
        self.h2h_history[key].append((date, result, home_team, gd))

        # 更新最近比赛日期
        self.last_match_date[home_team] = date
        self.last_match_date[away_team] = date


def prepare_dataset(matches_df: pd.DataFrame, start_year: int = 2002) -> Tuple[pd.DataFrame, EloRatingSystem]:
    """
    准备训练数据集
    返回：(特征DataFrame, 最终的Elo系统)
    """
    # 过滤时间范围
    df = matches_df[matches_df['date'].dt.year >= start_year].copy()
    df = df.sort_values('date').reset_index(drop=True)

    builder = MatchFeatureBuilder()

    # 第一阶段：用更早的数据（1872到start_year-1）预热Elo系统
    # 这样start_year开始的球队Elo已经积累了一些信息
    historical = matches_df[matches_df['date'].dt.year < start_year].sort_values('date')
    print(f"[准备] 用 {len(historical)} 场历史比赛预热Elo系统 (1872-{start_year-1})")
    for _, m in historical.iterrows():
        builder.update_with_match(
            m['home_team'], m['away_team'],
            m['home_score'], m['away_score'],
            m['date'], m['tournament'], m['neutral']
        )

    # 第二阶段：对 start_year 之后的比赛构建特征
    print(f"[构建] 为 {len(df)} 场比赛 ({start_year}-2025) 构建特征...")
    all_features = []
    all_labels = []
    all_meta = []

    for i, (_, m) in enumerate(df.iterrows()):
        # 构建特征（基于历史）
        feat = builder.build_features_for_match(
            m['home_team'], m['away_team'],
            m['date'], m['tournament'], m['neutral']
        )
        # 标签
        result = 1 if m['home_score'] > m['away_score'] else (0 if m['home_score'] == m['away_score'] else -1)
        all_features.append(feat)
        all_labels.append({
            'result': result,
            'home_score': int(m['home_score']),
            'away_score': int(m['away_score']),
            'total_goals': int(m['home_score'] + m['away_score']),
            'goal_diff': int(m['home_score'] - m['away_score']),
            'over_2_5': int(m['home_score'] + m['away_score'] > 2.5),
        })
        all_meta.append({
            'date': m['date'],
            'home_team': m['home_team'],
            'away_team': m['away_team'],
            'tournament': m['tournament'],
            'neutral': m['neutral'],
        })

        # 用当前比赛更新状态（必须放到特征构建之后！）
        builder.update_with_match(
            m['home_team'], m['away_team'],
            m['home_score'], m['away_score'],
            m['date'], m['tournament'], m['neutral']
        )

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1}/{len(df)} 场")

    features_df = pd.DataFrame(all_features)
    labels_df = pd.DataFrame(all_labels)
    meta_df = pd.DataFrame(all_meta)
    print(f"[完成] 特征矩阵: {features_df.shape}")
    print(f"[特征列] {list(features_df.columns)}")
    return features_df, labels_df, meta_df, builder


def get_elo_at_date(matches_df: pd.DataFrame, target_date, teams: List[str] = None) -> Dict[str, float]:
    """
    获取截至target_date的球队Elo等级分
    用于实时预测
    """
    builder = MatchFeatureBuilder()
    past = matches_df[matches_df['date'] < target_date].sort_values('date')
    for _, m in past.iterrows():
        if pd.isna(m['home_score']) or pd.isna(m['away_score']):
            continue
        builder.update_with_match(
            m['home_team'], m['away_team'],
            int(m['home_score']), int(m['away_score']),
            m['date'], m['tournament'], m['neutral']
        )
    if teams:
        return {t: builder.elo.get_rating(t) for t in teams}
    return builder.elo.to_dict()


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, '.')
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'results.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df['neutral'] = df['neutral'].map({'TRUE': True, 'FALSE': False, True: True, False: False}).fillna(True)

    # 只用最近几年数据测试
    test_df = df[df['date'].dt.year >= 2024].head(100)
    full_df = df[df['date'].dt.year >= 2000]
    features, labels, meta, builder = prepare_dataset(full_df, start_year=2020)
    print("\n样本特征:")
    print(features.head(2).T)
    print(f"\n标签分布:")
    print(labels['result'].value_counts(normalize=True))
