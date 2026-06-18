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
        self.feature_columns = None
        self.elo_ratings = None
        self.matches_df = None
        self.feature_builder = None  # 用于实时预测
        self.metadata = None
        self.version = version
        self._version_manager = ModelVersionManager(MODEL_DIR)
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
                date: datetime = None) -> dict:
        """
        预测一场比赛
        返回：胜平负概率、预测比分、解释、胜率
        """
        if date is None:
            date = datetime.now()

        # 构建特征
        features = self.feature_builder.build_features_for_match(
            home_team, away_team, date, tournament, neutral
        )

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

    def get_full_schedule(self):
        """返回 2026 世界杯完整赛程（小组赛 + 淘汰赛），含开球时间和时区"""
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

        # 构建 (date, home_team, away_team) → scores 映射
        score_map = {}
        for _, m in wc.iterrows():
            has_result = pd.notna(m['home_score']) and pd.notna(m['away_score'])
            key = (m['date'].strftime('%Y-%m-%d'), m['home_team'], m['away_team'])
            score_map[key] = {
                'home_score': int(m['home_score']) if has_result else None,
                'away_score': int(m['away_score']) if has_result else None,
                'played': has_result,
            }

        all_matches = []
        for m in sched_data['matches']:
            key = (m['date'], m['home_team'], m['away_team'])
            score_info = score_map.get(key, {'home_score': None, 'away_score': None, 'played': False})

            home_team = m['home_team']
            away_team = m['away_team']
            all_matches.append({
                'date': m['datetime'],  # 完整 ISO 时间（含时区偏移）
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
