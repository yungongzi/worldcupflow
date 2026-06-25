"""
预测器模块
加载训练好的模型，提供预测接口
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.features import (
    MatchFeatureBuilder, EloRatingSystem, get_tournament_weight
)
from model.version_manager import ModelVersionManager
from translations.teams_zh import get_chinese_name, TEAM_ZH

MODEL_DIR = Path(__file__).parent.parent / 'model' / 'saved'
DATA_DIR = Path(__file__).parent.parent / 'data'


class WorldCupPredictor:
    """世界杯预测器"""

    def __init__(self, version: str = None):
        """
        Args:
            version: 指定加载的模型版本ID。None 则加载根目录的活跃版本文件。
        """
        self.outcome_model = None
        self.home_score_model = None
        self.away_score_model = None
        self.agg_outcome_model = None
        self.agg_home_score_model = None
        self.agg_away_score_model = None
        self.feature_columns = None
        self.elo_ratings = None
        self.matches_df = None
        self.feature_builder = None  # 用于实时预测
        self.metadata = None
        self.version = version
        self._version_manager = ModelVersionManager(MODEL_DIR)
        self.has_aggressive = False  # 标记激进模型是否可用
        self.load()

    @property
    def _load_dir(self):
        """模型文件加载目录：指定版本则从版本目录加载，否则从根目录（活跃版本）"""
        if self.version:
            d = MODEL_DIR / 'versions' / self.version
            if d.exists():
                return d
            print(f"[警告] 版本 {self.version} 目录不存在，回退到根目录")
        return MODEL_DIR

    def load(self):
        """加载模型和数据"""
        load_dir = self._load_dir
        ver_label = self.version or self._version_manager.get_active_version_id() or "（根目录）"
        print(f"[加载] 加载模型... 版本: {ver_label}")
        # 加载XGBoost模型
        self.outcome_model = xgb.XGBClassifier()
        self.outcome_model.load_model(load_dir / 'outcome_model.json')

        self.home_score_model = xgb.XGBRegressor()
        self.home_score_model.load_model(load_dir / 'home_score_model.json')

        self.away_score_model = xgb.XGBRegressor()
        self.away_score_model.load_model(load_dir / 'away_score_model.json')

        # 尝试加载激进模型（如果存在）
        agg_outcome_path = load_dir / 'aggressive_outcome_model.json'
        agg_home_path = load_dir / 'aggressive_home_score_model.json'
        agg_away_path = load_dir / 'aggressive_away_score_model.json'
        if agg_outcome_path.exists() and agg_home_path.exists() and agg_away_path.exists():
            self.agg_outcome_model = xgb.XGBClassifier()
            self.agg_outcome_model.load_model(agg_outcome_path)
            self.agg_home_score_model = xgb.XGBRegressor()
            self.agg_home_score_model.load_model(agg_home_path)
            self.agg_away_score_model = xgb.XGBRegressor()
            self.agg_away_score_model.load_model(agg_away_path)
            self.has_aggressive = True
            print(f"[加载] 激进模型已加载 (Tweedie + 放松正则化)")
        else:
            print(f"[加载] 未找到激进模型，predict_both 将回退到启发式调整")

        # 特征列
        with open(load_dir / 'feature_columns.json', 'r', encoding='utf-8') as f:
            self.feature_columns = json.load(f)

        # Elo等级分
        with open(load_dir / 'elo_ratings.json', 'r', encoding='utf-8') as f:
            self.elo_ratings = json.load(f)

        # 训练元数据
        with open(load_dir / 'training_metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # 加载历史比赛数据
        df = pd.read_csv(DATA_DIR / 'results.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        df['neutral'] = df['neutral'].astype(str).map({'TRUE': True, 'FALSE': False, 'True': True, 'False': False}).fillna(True)
        self.matches_df = df.sort_values('date').reset_index(drop=True)

        # 构建feature builder，并预热到最新日期
        self.feature_builder = MatchFeatureBuilder()
        # 用所有已完成的比赛更新状态
        completed = df.dropna(subset=['home_score', 'away_score'])
        print(f"[加载] 用 {len(completed)} 场历史比赛初始化状态...")
        for _, m in completed.iterrows():
            self.feature_builder.update_with_match(
                m['home_team'], m['away_team'],
                int(m['home_score']), int(m['away_score']),
                m['date'], m['tournament'], m['neutral']
            )
        print(f"[加载] 完成. 共 {len(self.elo_ratings)} 支球队有Elo等级分")

        # 计算 WC2026 赛事统计（用于激进模式调整）
        self._compute_tournament_stats()

    def _compute_tournament_stats(self):
        """计算 WC2026 赛事统计数据：场均进球、各队偏差"""
        wc2026 = self.matches_df[
            (self.matches_df['tournament'] == 'FIFA World Cup') &
            (self.matches_df['date'].dt.year >= 2026) &
            self.matches_df['home_score'].notna()
        ].copy()
        wc2026['home_score'] = wc2026['home_score'].astype(int)
        wc2026['away_score'] = wc2026['away_score'].astype(int)

        if len(wc2026) >= 3:
            self._wc2026_avg_goals = float((wc2026['home_score'] + wc2026['away_score']).mean())
        else:
            self._wc2026_avg_goals = 2.8  # 默认值：世界杯历史均值偏上

        # 历史场均进球（所有数据）
        all_with_scores = self.matches_df.dropna(subset=['home_score', 'away_score'])
        all_with_scores.loc[:, 'home_score'] = all_with_scores['home_score'].astype(int)
        all_with_scores.loc[:, 'away_score'] = all_with_scores['away_score'].astype(int)
        self._historical_avg_goals = float((all_with_scores['home_score'] + all_with_scores['away_score']).mean()) if len(all_with_scores) > 0 else 2.6

        # 各队 WC2026 进球/失球统计
        self._wc2026_team_goals = {}
        for _, m in wc2026.iterrows():
            home = m['home_team']
            away = m['away_team']
            hs = int(m['home_score'])
            aws = int(m['away_score'])
            # 主队
            if home not in self._wc2026_team_goals:
                self._wc2026_team_goals[home] = {'gf': 0, 'ga': 0, 'games': 0}
            self._wc2026_team_goals[home]['gf'] += hs
            self._wc2026_team_goals[home]['ga'] += aws
            self._wc2026_team_goals[home]['games'] += 1
            # 客队
            if away not in self._wc2026_team_goals:
                self._wc2026_team_goals[away] = {'gf': 0, 'ga': 0, 'games': 0}
            self._wc2026_team_goals[away]['gf'] += aws
            self._wc2026_team_goals[away]['ga'] += hs
            self._wc2026_team_goals[away]['games'] += 1

        # 各队历史场均进球（从 matches_df 汇总）
        self._team_historical_avg = {}
        for team in set(all_with_scores['home_team'].unique()) | set(all_with_scores['away_team'].unique()):
            home_mask = all_with_scores['home_team'] == team
            away_mask = all_with_scores['away_team'] == team
            total_gf = int(all_with_scores.loc[home_mask, 'home_score'].sum()) + int(all_with_scores.loc[away_mask, 'away_score'].sum())
            total_games = home_mask.sum() + away_mask.sum()
            if total_games > 0:
                self._team_historical_avg[team] = total_gf / total_games

        # 通胀因子：WC2026 场均进球 / 历史场均进球
        self._goal_inflation = max(1.0, self._wc2026_avg_goals / max(self._historical_avg_goals, 1.0))
        print(f"[加载] 赛事统计: WC2026 场均={self._wc2026_avg_goals:.2f}球, "
              f"历史均值={self._historical_avg_goals:.2f}球, 通胀因子={self._goal_inflation:.2f}x")

    def _get_team_wc2026_scoring_rate(self, team: str) -> float:
        """获取球队在 WC2026 的场均进球率"""
        if team in self._wc2026_team_goals:
            stats = self._wc2026_team_goals[team]
            return stats['gf'] / max(stats['games'], 1)
        return 0.0

    def _get_team_historical_scoring_rate(self, team: str) -> float:
        """获取球队历史场均进球率"""
        return self._team_historical_avg.get(team, 1.2)

    def _get_team_form_deviation(self, team: str) -> float:
        """
        计算球队近期进球偏差:
        >1: WC2026 比历史更猛 (激进上修)
        <1: WC2026 比历史更弱 (保持原样)
        """
        wc26_rate = self._get_team_wc2026_scoring_rate(team)
        hist_rate = self._get_team_historical_scoring_rate(team)
        if wc26_rate > 0 and hist_rate > 0:
            deviation = wc26_rate / hist_rate
            return max(0.8, min(deviation, 2.5))  # clamp到 [0.8, 2.5]
        return 1.0  # 没有 WC2026 数据，用 1.0 (不调整)

    def reload(self, version: str = None):
        """
        重新加载模型（切换版本）。
        Args:
            version: 目标版本ID。None 则加载根目录的活跃版本文件。
        """
        self.version = version
        self.load()

    def get_current_version(self) -> str:
        """获取当前加载的版本ID"""
        return self.version or self._version_manager.get_active_version_id() or "unknown"

    def predict(self, home_team: str, away_team: str,
                tournament: str = 'FIFA World Cup',
                neutral: bool = True,
                date: datetime = None,
                mode: str = 'conservative') -> dict:
        """
        预测一场比赛
        Args:
            mode: 'conservative' (默认, 保守) 或 'aggressive' (激进)
        返回：胜平负概率、预测比分、解释、胜率
        """
        if mode == 'aggressive':
            return self._predict_aggressive(home_team, away_team, tournament, neutral, date)
        return self._predict_conservative(home_team, away_team, tournament, neutral, date)

    def predict_both(self, home_team: str, away_team: str,
                     tournament: str = 'FIFA World Cup',
                     neutral: bool = True,
                     date: datetime = None) -> dict:
        """同时返回保守和激进两种预测"""
        conservative = self._predict_conservative(home_team, away_team, tournament, neutral, date)
        aggressive = self._predict_aggressive(home_team, away_team, tournament, neutral, date)
        return {
            'home_team': conservative['home_team'],
            'away_team': conservative['away_team'],
            'home_team_zh': conservative['home_team_zh'],
            'away_team_zh': conservative['away_team_zh'],
            'conservative': {
                'probabilities': conservative['probabilities'],
                'predicted_score': conservative['predicted_score'],
                'top_scores': conservative['top_scores'],
                'win_rates': conservative['win_rates'],
                'elo_ratings': conservative['elo_ratings'],
                'explanation': conservative['explanation'],
            },
            'aggressive': {
                'probabilities': aggressive['probabilities'],
                'predicted_score': aggressive['predicted_score'],
                'top_scores': aggressive['top_scores'],
                'win_rates': aggressive['win_rates'],
                'elo_ratings': aggressive['elo_ratings'],
                'explanation': aggressive['explanation'],
            },
            'tournament': tournament,
            'neutral': neutral,
            'predict_time': datetime.now().isoformat(),
            'adjustment_factors': {
                'tournament_inflation': round(self._goal_inflation, 2),
                'wc2026_avg_goals': round(self._wc2026_avg_goals, 2),
                'historical_avg_goals': round(self._historical_avg_goals, 2),
                'home_form_deviation': round(self._get_team_form_deviation(home_team), 2),
                'away_form_deviation': round(self._get_team_form_deviation(away_team), 2),
            },
        }

    def _predict_conservative(self, home_team: str, away_team: str,
                              tournament: str = 'FIFA World Cup',
                              neutral: bool = True,
                              date: datetime = None) -> dict:
        """
        保守预测（原始模型，不做任何调整）
        返回：胜平负概率、预测比分、解释、胜率
        """
        if date is None:
            date = datetime.now()

        # 构建特征
        features = self._build_feature_vector(home_team, away_team, tournament, neutral, date)

        # 转成DataFrame（按训练时的列顺序）
        feat_df = pd.DataFrame([features])
        # 添加缺失的列（默认0）
        for col in self.feature_columns:
            if col not in feat_df.columns:
                feat_df[col] = 0
        feat_df = feat_df[self.feature_columns]

        # 1. 胜平负概率
        outcome_prob = self.outcome_model.predict_proba(feat_df)[0]
        # outcome_model的标签: 0=客胜, 1=平, 2=主胜
        prob_away_win = float(outcome_prob[0])
        prob_draw = float(outcome_prob[1])
        prob_home_win = float(outcome_prob[2])

        # 2. 进球数预测
        home_goals_pred = float(self.home_score_model.predict(feat_df)[0])
        away_goals_pred = float(self.away_score_model.predict(feat_df)[0])
        home_goals = max(0, int(round(home_goals_pred)))
        away_goals = max(0, int(round(away_goals_pred)))

        # 3. 比分概率矩阵（基于Poisson）
        score_matrix = self._compute_score_probability_matrix(
            home_goals_pred, away_goals_pred
        )
        # Top 5最可能比分
        top_scores = self._top_scores(score_matrix, n=5)

        # 4. 解释
        explanation = self._generate_explanation(
            home_team, away_team, features, prob_home_win, prob_draw, prob_away_win,
            home_goals_pred, away_goals_pred, top_scores
        )

        # 5. 胜率（主胜+0.5*平局）
        home_win_rate = prob_home_win + 0.5 * prob_draw
        away_win_rate = prob_away_win + 0.5 * prob_draw

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_team_zh': get_chinese_name(home_team),
            'away_team_zh': get_chinese_name(away_team),
            'probabilities': {
                'home_win': round(prob_home_win, 4),
                'draw': round(prob_draw, 4),
                'away_win': round(prob_away_win, 4),
            },
            'predicted_score': {
                'home': home_goals,
                'away': away_goals,
                'home_expected': round(home_goals_pred, 2),
                'away_expected': round(away_goals_pred, 2),
            },
            'top_scores': top_scores,
            'win_rates': {
                'home': round(home_win_rate, 4),
                'away': round(away_win_rate, 4),
            },
            'elo_ratings': {
                'home': round(self.elo_ratings.get(home_team, 1500), 1),
                'away': round(self.elo_ratings.get(away_team, 1500), 1),
                'diff': round(self.elo_ratings.get(home_team, 1500) - self.elo_ratings.get(away_team, 1500), 1),
            },
            'features': {
                'home_form_winrate': round(features.get('home_form_10_winrate', 0.5), 3),
                'away_form_winrate': round(features.get('away_form_10_winrate', 0.5), 3),
                'home_form_goals_scored': round(features.get('home_form_10_goals_scored_avg', 1.2), 2),
                'away_form_goals_scored': round(features.get('away_form_10_goals_scored_avg', 1.2), 2),
                'h2h_count': features.get('h2h_count', 0),
                'h2h_home_winrate': round(features.get('h2h_home_winrate', 0.5), 3),
            },
            'explanation': explanation,
            'tournament': tournament,
            'neutral': neutral,
            'predict_time': datetime.now().isoformat(),
        }

    def _compute_score_probability_matrix(self, lambda_home, lambda_away, max_goals=10):
        """基于Poisson分布计算比分概率矩阵"""
        from scipy.stats import poisson
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                matrix[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
        # 归一化
        matrix = matrix / matrix.sum()
        return matrix

    def _top_scores(self, matrix, n=5):
        """获取Top N最可能的比分"""
        scores = []
        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):
                scores.append((f"{h}-{a}", h, a, matrix[h, a]))
        scores.sort(key=lambda x: x[3], reverse=True)
        return [
            {
                'score': s[0],
                'home': s[1],
                'away': s[2],
                'probability': round(float(s[3]), 4),
            }
            for s in scores[:n]
        ]

    def _generate_explanation(self, home_team, away_team, features,
                              p_home, p_draw, p_away, hg_pred, ag_pred, top_scores):
        """生成预测解释（中文）"""
        home_zh = get_chinese_name(home_team)
        away_zh = get_chinese_name(away_team)
        elo_h = self.elo_ratings.get(home_team, 1500)
        elo_a = self.elo_ratings.get(away_team, 1500)
        elo_diff = elo_h - elo_a

        parts = []

        # 1. 总体判断
        if p_home > p_away + 0.1:
            favorite = home_zh
            fav_prob = p_home
            fav_elo = elo_h
            underdog_elo = elo_a
        elif p_away > p_home + 0.1:
            favorite = away_zh
            fav_prob = p_away
            fav_elo = elo_a
            underdog_elo = elo_h
        else:
            favorite = None

        if favorite:
            advantage = abs(elo_diff)
            if advantage > 200:
                strength = "实力明显占优"
            elif advantage > 100:
                strength = "实力较强"
            else:
                strength = "略占优势"
            parts.append(f"📊 {favorite}（Elo {fav_elo:.0f}）{strength}（领先{abs(elo_diff):.0f}分），胜率达到 {fav_prob*100:.1f}%")
        else:
            parts.append(f"⚖️ 双方实力接近（Elo差距仅 {abs(elo_diff):.0f} 分），胜负难料")

        # 2. 近期状态
        home_form = features.get('home_form_10_winrate', 0.5)
        away_form = features.get('away_form_10_winrate', 0.5)
        home_gd = features.get('home_form_10_goaldiff_avg', 0)
        away_gd = features.get('away_form_10_goaldiff_avg', 0)
        if home_form > 0.6:
            parts.append(f"🔥 {home_zh}近期状态火热（近10场胜率 {home_form*100:.0f}%，场均净胜 {home_gd:+.1f}球）")
        elif home_form < 0.3:
            parts.append(f"❄️ {home_zh}近期状态低迷（近10场胜率仅 {home_form*100:.0f}%）")
        if away_form > 0.6:
            parts.append(f"🔥 {away_zh}近期状态火热（近10场胜率 {away_form*100:.0f}%，场均净胜 {away_gd:+.1f}球）")
        elif away_form < 0.3:
            parts.append(f"❄️ {away_zh}近期状态低迷（近10场胜率仅 {away_form*100:.0f}%）")

        # 3. 攻防分析
        home_attack = features.get('home_form_10_goals_scored_avg', 1.2)
        away_attack = features.get('away_form_10_goals_scored_avg', 1.2)
        home_def = features.get('home_form_10_goals_conceded_avg', 1.2)
        away_def = features.get('away_form_10_goals_conceded_avg', 1.2)
        if home_attack > 2.0 and away_def > 1.5:
            parts.append(f"⚔️ {home_zh}攻击力强（场均{home_attack:.1f}球）vs {away_zh}防守薄弱（场均丢{away_def:.1f}球），主队有望破门多次")
        if away_attack > 2.0 and home_def > 1.5:
            parts.append(f"⚔️ {away_zh}攻击力强（场均{away_attack:.1f}球）vs {home_zh}防守薄弱（场均丢{home_def:.1f}球），客队反击有威胁")

        # 4. 对战记录
        h2h_count = features.get('h2h_count', 0)
        h2h_home_wr = features.get('h2h_home_winrate', 0.5)
        if h2h_count >= 3:
            if h2h_home_wr > 0.6:
                parts.append(f"📜 历史交锋 {h2h_count} 次，{home_zh}占据上风（胜率 {h2h_home_wr*100:.0f}%）")
            elif h2h_home_wr < 0.4:
                parts.append(f"📜 历史交锋 {h2h_count} 次，{away_zh}占据上风（{home_zh}胜率仅 {h2h_home_wr*100:.0f}%）")
            else:
                parts.append(f"📜 历史交锋 {h2h_count} 次，互有胜负")

        # 5. 预测比分说明
        top_score = top_scores[0]
        parts.append(f"🎯 预测最可能比分 {top_score['score']}（概率 {top_score['probability']*100:.1f}%），主队期望进球 {hg_pred:.2f}，客队期望进球 {ag_pred:.2f}")

        return parts

    def _build_feature_vector(self, home_team: str, away_team: str,
                               tournament: str = 'FIFA World Cup',
                               neutral: bool = True,
                               date: datetime = None) -> dict:
        """构建特征向量（共享方法，保守和激进模型都用）"""
        if date is None:
            date = datetime.now()
        return self.feature_builder.build_features_for_match(
            home_team, away_team, date, tournament, neutral
        )

    def _predict_aggressive(self, home_team: str, away_team: str,
                            tournament: str = 'FIFA World Cup',
                            neutral: bool = True,
                            date: datetime = None) -> dict:
        """激进预测 — 使用 Tweedie + 放松正则化的独立模型推理"""
        if not self.has_aggressive:
            # 回退到启发式调整（旧行为）
            return self._predict_aggressive_heuristic(home_team, away_team, tournament, neutral, date)

        # 构建特征向量（与保守模型共享）
        features = self._build_feature_vector(home_team, away_team, tournament, neutral, date)

        # 构建 DataFrame 用于 XGBoost 推理
        X = pd.DataFrame([features])[self.feature_columns]

        # 激进胜平负预测
        outcome_proba = self.agg_outcome_model.predict_proba(X)[0]
        # XGBoost 输出 [客胜, 平, 主胜] 对应 label 0,1,2
        away_win_prob = float(outcome_proba[0])
        draw_prob = float(outcome_proba[1])
        home_win_prob = float(outcome_proba[2])

        # 激进进球期望
        home_lambda = float(max(0, self.agg_home_score_model.predict(X)[0]))
        away_lambda = float(max(0, self.agg_away_score_model.predict(X)[0]))

        # 计算比分概率矩阵（Tweedie 给期望值，仍需 Poisson 矩阵算分布）
        score_matrix = self._compute_score_probability_matrix(home_lambda, away_lambda)
        top_scores = self._top_scores(score_matrix, n=5)

        # 预测比分（期望值取整）
        home_goals = max(0, int(round(home_lambda)))
        away_goals = max(0, int(round(away_lambda)))

        # 综合胜率
        home_win_rate = home_win_prob + 0.5 * draw_prob
        away_win_rate = away_win_prob + 0.5 * draw_prob

        # 解释（包含激进模型特有信息）
        explanation = self._generate_aggressive_explanation_v2(
            home_team, away_team, features, home_win_prob, draw_prob, away_win_prob,
            home_lambda, away_lambda, top_scores, tournament
        )

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_team_zh': get_chinese_name(home_team),
            'away_team_zh': get_chinese_name(away_team),
            'probabilities': {
                'home_win': round(home_win_prob, 4),
                'draw': round(draw_prob, 4),
                'away_win': round(away_win_prob, 4),
            },
            'predicted_score': {
                'home': home_goals,
                'away': away_goals,
                'home_expected': round(home_lambda, 2),
                'away_expected': round(away_lambda, 2),
            },
            'top_scores': top_scores,
            'win_rates': {
                'home': round(home_win_rate, 4),
                'away': round(away_win_rate, 4),
            },
            'elo_ratings': {
                'home': round(features['elo_home'], 1),
                'away': round(features['elo_away'], 1),
                'diff': round(features['elo_diff'], 1),
            },
            'features': features,
            'explanation': explanation,
            'tournament': tournament,
            'neutral': neutral,
            'predict_time': datetime.now().isoformat(),
            'model_type': 'tweedie_aggressive',
            'adjustment_factors': {
                'tournament_avg_goals': round(features.get('tournament_avg_goals', 2.6), 2),
                'home_tournament_attack_bias': round(features.get('home_tournament_attack_bias', 1.0), 2),
                'away_tournament_attack_bias': round(features.get('away_tournament_attack_bias', 1.0), 2),
            },
        }

    def _predict_aggressive_heuristic(self, home_team: str, away_team: str,
                                       tournament: str = 'FIFA World Cup',
                                       neutral: bool = True,
                                       date: datetime = None) -> dict:
        """激进预测回退方案 — 基于保守预测 + 赛事走势调整（无独立激进模型时使用）"""
        # 1. 先获取保守预测作为基准
        base = self._predict_conservative(home_team, away_team, tournament, neutral, date)

        # 2. 计算调整因子
        inflation = self._goal_inflation
        home_dev = self._get_team_form_deviation(home_team)
        away_dev = self._get_team_form_deviation(away_team)

        # 3. 调整期望进球
        lambda_h_base = base['predicted_score']['home_expected']
        lambda_a_base = base['predicted_score']['away_expected']

        lambda_h_adj = lambda_h_base * inflation * home_dev
        lambda_a_adj = lambda_a_base * inflation * away_dev

        lambda_h_agg = 0.4 * lambda_h_base + 0.6 * max(lambda_h_base, lambda_h_adj)
        lambda_a_agg = 0.4 * lambda_a_base + 0.6 * max(lambda_a_base, lambda_a_adj)

        lambda_h_agg = min(lambda_h_agg, lambda_h_base * 2.5)
        lambda_a_agg = min(lambda_a_agg, lambda_a_base * 2.5)

        # 4. 调整胜平负概率
        home_win = base['probabilities']['home_win']
        draw = base['probabilities']['draw']
        away_win = base['probabilities']['away_win']

        draw_discount = 0.7
        draw_adj = draw * draw_discount
        freed_prob = draw - draw_adj

        if home_win > away_win:
            advantage_ratio = home_win / max(home_win + away_win, 0.01)
        else:
            advantage_ratio = 1 - (home_win / max(home_win + away_win, 0.01))
        home_win_adj = home_win + freed_prob * advantage_ratio
        away_win_adj = away_win + freed_prob * (1 - advantage_ratio)

        # 5. 计算比分
        home_goals_agg = max(0, int(round(lambda_h_agg)))
        away_goals_agg = max(0, int(round(lambda_a_agg)))

        agg_matrix = self._compute_score_probability_matrix(lambda_h_agg, lambda_a_agg)
        top_scores_agg = self._top_scores(agg_matrix, n=5)

        home_win_rate_agg = home_win_adj + 0.5 * draw_adj
        away_win_rate_agg = away_win_adj + 0.5 * draw_adj

        explanation_agg = self._generate_aggressive_explanation_v2(
            home_team, away_team, base['features'],
            home_win_adj, draw_adj, away_win_adj,
            lambda_h_agg, lambda_a_agg, top_scores_agg, tournament
        )
        explanation_agg.append(f"💡 [回退模式] 未训练激进模型，使用赛事通胀因子 {inflation:.2f}x + 球队形态偏差调整")

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_team_zh': get_chinese_name(home_team),
            'away_team_zh': get_chinese_name(away_team),
            'probabilities': {
                'home_win': round(home_win_adj, 4),
                'draw': round(draw_adj, 4),
                'away_win': round(away_win_adj, 4),
            },
            'predicted_score': {
                'home': home_goals_agg,
                'away': away_goals_agg,
                'home_expected': round(lambda_h_agg, 2),
                'away_expected': round(lambda_a_agg, 2),
            },
            'top_scores': top_scores_agg,
            'win_rates': {
                'home': round(home_win_rate_agg, 4),
                'away': round(away_win_rate_agg, 4),
            },
            'elo_ratings': base['elo_ratings'],
            'features': base['features'],
            'explanation': explanation_agg,
            'tournament': tournament,
            'neutral': neutral,
            'predict_time': datetime.now().isoformat(),
            'model_type': 'heuristic_aggressive',
            'adjustment_factors': {
                'tournament_inflation': round(inflation, 2),
                'home_form_deviation': round(home_dev, 2),
                'away_form_deviation': round(away_dev, 2),
                'lambda_h_original': round(lambda_h_base, 2),
                'lambda_a_original': round(lambda_a_base, 2),
                'lambda_h_adjusted': round(lambda_h_agg, 2),
                'lambda_a_adjusted': round(lambda_a_agg, 2),
            },
        }

    def _generate_aggressive_explanation_v2(self, home_team, away_team, features,
                                             home_win, draw, away_win,
                                             home_lambda, away_lambda, top_scores,
                                             tournament):
        """生成激进预测解释（v2: 基于真实 Tweedie 模型）"""
        home_zh = get_chinese_name(home_team)
        away_zh = get_chinese_name(away_team)
        parts = []

        parts.append(f"🔥【激进模式】Tweedie 模型 + 赛事走势特征")

        # 赛事走势
        tour_avg = features.get('tournament_avg_goals', 2.6)
        tour_over_rate = features.get('tournament_over_2_5_rate', 0.5)
        if features.get('tournament_match_count', 0) >= 3:
            parts.append(f"📊 {tournament} 场均 {tour_avg:.1f} 球，大2.5率 {tour_over_rate*100:.0f}%")

        # 球队赛事火力偏差
        home_bias = features.get('home_tournament_attack_bias', 1.0)
        away_bias = features.get('away_tournament_attack_bias', 1.0)
        if home_bias > 1.15:
            parts.append(f"⚡ {home_zh} 本届进球率比生涯高 {((home_bias-1)*100):.0f}%")
        if away_bias > 1.15:
            parts.append(f"⚡ {away_zh} 本届进球率比生涯高 {((away_bias-1)*100):.0f}%")

        # 概率
        parts.append(f"🎯 胜平负: {home_zh} {home_win*100:.1f}% | 平 {draw*100:.1f}% | {away_zh} {away_win*100:.1f}%")

        # 期望进球
        parts.append(f"🥅 期望进球: {home_zh} {home_lambda:.2f} vs {away_zh} {away_lambda:.2f}")

        # 最可能比分
        if top_scores:
            parts.append(f"📊 最可能比分: {top_scores[0]['score']}（{top_scores[0]['probability']*100:.1f}%）")

        return parts

    def get_team_list(self, top_n=80):
        """获取球队列表（按Elo排序）"""
        teams = sorted(self.elo_ratings.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                'team_en': t,
                'team_zh': get_chinese_name(t),
                'elo': round(r, 1),
            }
            for t, r in teams[:top_n]
        ]

    def search_team(self, query: str):
        """模糊搜索球队（支持中英文）"""
        query = query.strip().lower()
        results = []
        for team_en, elo in self.elo_ratings.items():
            team_zh = get_chinese_name(team_en)
            if query in team_en.lower() or query in team_zh:
                results.append({
                    'team_en': team_en,
                    'team_zh': team_zh,
                    'elo': round(elo, 1),
                })
        results.sort(key=lambda x: x['elo'], reverse=True)
        return results[:20]

    def get_upcoming_world_cup_matches(self, limit=50):
        """获取即将到来的世界杯比赛（数据中NA比分的）"""
        wc = self.matches_df[
            (self.matches_df['tournament'] == 'FIFA World Cup') &
            (self.matches_df['home_score'].isna())
        ].sort_values('date')
        return [
            {
                'date': m['date'].isoformat(),
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'home_team_zh': get_chinese_name(m['home_team']),
                'away_team_zh': get_chinese_name(m['away_team']),
                'city': m['city'],
                'country': m['country'],
            }
            for _, m in wc.head(limit).iterrows()
        ]

    def get_recent_results(self, limit=20):
        """获取最近的比赛结果"""
        completed = self.matches_df.dropna(subset=['home_score', 'away_score'])
        recent = completed.sort_values('date', ascending=False).head(limit)
        return [
            {
                'date': m['date'].isoformat(),
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'home_team_zh': get_chinese_name(m['home_team']),
                'away_team_zh': get_chinese_name(m['away_team']),
                'home_score': int(m['home_score']),
                'away_score': int(m['away_score']),
                'tournament': m['tournament'],
                'result': 'home_win' if m['home_score'] > m['away_score'] else ('draw' if m['home_score'] == m['away_score'] else 'away_win'),
            }
            for _, m in recent.iterrows()
        ]

    # 2026 世界杯分组映射
    WC2026_GROUPS = {
        'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
        'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
        'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
        'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
        'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
        'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
        'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
        'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
        'I': ['France', 'Senegal', 'Iraq', 'Norway'],
        'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
        'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
        'L': ['England', 'Croatia', 'Ghana', 'Panama'],
    }

    def _build_team_group_map(self):
        """构建 team -> group 映射"""
        m = {}
        for group, teams in self.WC2026_GROUPS.items():
            for t in teams:
                m[t] = group
        return m

    # 队名归一化映射（处理 ESPN/CSV 与赛程 JSON 中的拼写差异）
    _TEAM_NAME_ALIASES = {
        "Côte d'Ivoire": "Ivory Coast",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
        "Korea Republic": "South Korea",
        "Cape Verde Islands": "Cape Verde",
        "United Korea Republic": "South Korea",
    }

    @classmethod
    def _normalize_team(cls, name: str) -> str:
        """归一化球队名称"""
        if not name or name == 'TBD':
            return name
        return cls._TEAM_NAME_ALIASES.get(name, name)

    @classmethod
    def _build_score_lookup_key(cls, date_str: str, home: str, away: str) -> str:
        """构建比分查找键（忽略队名大小写差异）"""
        n_home = cls._normalize_team(home).lower() if home else ''
        n_away = cls._normalize_team(away).lower() if away else ''
        return f"{date_str}|{n_home}|{n_away}"

    def get_full_schedule(self):
        """返回 2026 世界杯完整赛程（小组赛 + 淘汰赛），含开球时间和时区"""
        from datetime import timedelta

        # 读取赛程 JSON（包含所有 104 场比赛的开球时间、时区偏移、场馆）
        schedule_file = DATA_DIR / 'wc2026_schedule.json'
        if not schedule_file.exists():
            return []

        with open(schedule_file, 'r', encoding='utf-8') as f:
            sched_data = json.load(f)

        # 从 results.csv 获取已完成的比赛比分
        wc = self.matches_df[
            (self.matches_df['tournament'] == 'FIFA World Cup') &
            (self.matches_df['date'].dt.year >= 2026)
        ].sort_values('date')

        # 构建多层索引的 score_map：
        # 1) 精确匹配 (date, home, away)
        # 2) 归一化匹配 (date, norm_home, norm_away) 用于跨别名
        # 3) 归一化 + 互换匹配 (date, norm_away, norm_home) 用于主客互换
        # 4) 跨 ±1 天匹配（处理 UTC 时区导致的日期差）
        score_map = {}       # 精确匹配
        norm_score_map = {}  # 归一化匹配
        swap_score_map = {}  # 归一化互换匹配

        for _, m in wc.iterrows():
            has_result = pd.notna(m['home_score']) and pd.notna(m['away_score'])
            date_str = m['date'].strftime('%Y-%m-%d')
            home_team = m['home_team']
            away_team = m['away_team']

            info = {
                'home_score': int(m['home_score']) if has_result else None,
                'away_score': int(m['away_score']) if has_result else None,
                'played': has_result,
                'source_date': date_str,
            }

            # 精确 key
            score_map[(date_str, home_team, away_team)] = info
            # 归一化 key
            norm_key = (date_str, self._normalize_team(home_team), self._normalize_team(away_team))
            if norm_key not in norm_score_map or not norm_score_map[norm_key].get('played'):
                norm_score_map[norm_key] = info
            # 互换 key（主客队可能被 ESPN 反了）
            swap_key = (date_str, self._normalize_team(away_team), self._normalize_team(home_team))
            if swap_key not in swap_score_map:
                swap_score_map[swap_key] = {
                    'home_score': info['away_score'],
                    'away_score': info['home_score'],
                    'played': info['played'],
                    'source_date': date_str,
                }

        def lookup_score(sched_date, sched_home, sched_away):
            """多策略查找比分：
            1. 精确匹配
            2. 归一化匹配
            3. 归一化 + 主客互换
            4. 跨 ±2 天归一化匹配（处理时区差）
            """
            # 1. 精确
            key = (sched_date, sched_home, sched_away)
            if key in score_map:
                return score_map[key]

            # 2. 归一化
            norm_key = (sched_date, self._normalize_team(sched_home), self._normalize_team(sched_away))
            if norm_key in norm_score_map:
                return norm_score_map[norm_key]

            # 3. 归一化 + 互换
            swap_key = (sched_date, self._normalize_team(sched_away), self._normalize_team(sched_home))
            if swap_key in swap_score_map:
                return swap_score_map[swap_key]

            # 4. 跨 ±2 天（时区差 + 数据录入日期偏差）
            from datetime import datetime as dt
            try:
                base_date = dt.strptime(sched_date, '%Y-%m-%d')
            except ValueError:
                return {'home_score': None, 'away_score': None, 'played': False}

            for delta_days in [1, -1, 2, -2]:
                alt_date = (base_date + timedelta(days=delta_days)).strftime('%Y-%m-%d')
                # 归一化正向
                alt_key = (alt_date, self._normalize_team(sched_home), self._normalize_team(sched_away))
                if alt_key in norm_score_map and norm_score_map[alt_key].get('played'):
                    return norm_score_map[alt_key]
                # 归一化互换
                alt_swap_key = (alt_date, self._normalize_team(sched_away), self._normalize_team(sched_home))
                if alt_swap_key in swap_score_map and swap_score_map[alt_swap_key].get('played'):
                    return swap_score_map[alt_swap_key]

            return {'home_score': None, 'away_score': None, 'played': False}

        all_matches = []
        for m in sched_data['matches']:
            home_team = m['home_team']
            away_team = m['away_team']
            score_info = lookup_score(m['date'], home_team, away_team)

            all_matches.append({
                'date': m['datetime'],  # 完整 ISO 时间（含时区偏移）
                'match_id': m.get('match_id', 0),
                'phase': m['phase'],
                'phase_zh': m['phase_zh'],
                'group': m.get('group', ''),
                'home_team': home_team,
                'away_team': away_team,
                'home_team_zh': get_chinese_name(home_team) if home_team != 'TBD' else '待定',
                'away_team_zh': get_chinese_name(away_team) if away_team != 'TBD' else '待定',
                'home_score': score_info['home_score'],
                'away_score': score_info['away_score'],
                'city': m.get('city', ''),
                'country': m.get('country', ''),
                'stadium': m.get('stadium', ''),
                'played': score_info['played'],
            })

        all_matches.sort(key=lambda x: x['date'])
        return all_matches

    def get_h2h_records(self, team_a: str, team_b: str, limit: int = 5):
        """获取两队历史交锋记录"""
        df = self.matches_df.dropna(subset=['home_score', 'away_score'])
        mask = (
            ((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
            ((df['home_team'] == team_b) & (df['away_team'] == team_a))
        )
        h2h = df[mask].sort_values('date', ascending=False).head(limit)
        return [
            {
                'date': m['date'].isoformat(),
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'home_score': int(m['home_score']),
                'away_score': int(m['away_score']),
                'tournament': m['tournament'],
            }
            for _, m in h2h.iterrows()
        ]


if __name__ == "__main__":
    # 测试
    predictor = WorldCupPredictor()
    print("\n=== 测试预测 ===")
    result = predictor.predict("Brazil", "Argentina", "FIFA World Cup", True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
