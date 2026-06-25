"""
数据抓取模块
策略：
1. 主数据源：Kaggle数据（已包含到2026世界杯）
2. 补充数据源：Wikipedia等公开页面（抓取最新比分）
3. 兜底：使用本地缓存数据

注意：本模块设计为可扩展，支持未来添加更多数据源
"""
import os
import re
import json
import httpx
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / 'data'
CACHE_FILE = DATA_DIR / 'live_cache.json'


# ESPN队名 -> Kaggle数据中的标准队名
TEAM_NAME_NORMALIZATION = {
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Iran, Islamic Republic of": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "United States": "United States",
    "USA": "United States",
    "Republic of Ireland": "Ireland",
    "North Macedonia": "North Macedonia",
    "China PR": "China PR",
    "Chinese Taipei": "Chinese Taipei",
    "Congo": "Congo",
    "Cape Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herz.": "Bosnia and Herzegovina",
    "United Korea Republic": "South Korea",
}


def normalize_team_name(name: str) -> str:
    """归一化球队名称（与Kaggle数据保持一致）"""
    if name in TEAM_NAME_NORMALIZATION:
        return TEAM_NAME_NORMALIZATION[name]
    return name


async def fetch_wikipedia_world_cup_results(year: int = 2026) -> List[Dict]:
    """
    从Wikipedia抓取世界杯比赛结果
    优点：免费、稳定、覆盖全面
    缺点：HTML结构可能变化
    """
    url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup"
    results = []
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; WorldCupPredictor/1.0)'
            })
            if resp.status_code != 200:
                print(f"[Wikipedia] HTTP {resp.status_code}")
                return results
            soup = BeautifulSoup(resp.text, 'lxml')
            # 解析比赛表格（这个结构会随时间变化，需要灵活处理）
            # 这里只做示例，实际解析需要根据页面结构定制
            tables = soup.find_all('table', class_='fevent')
            for table in tables:
                try:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 5:
                            # 尝试解析队伍和比分
                            text = ' '.join(c.get_text(strip=True) for c in cells)
                            # 简化的解析逻辑
                            match = re.search(r'(\d+)\s*[-–]\s*(\d+)', text)
                            if match:
                                # 这里需要更复杂的解析逻辑
                                pass
                except Exception as e:
                    continue
    except Exception as e:
        print(f"[Wikipedia] 抓取失败: {e}")
    return results


async def fetch_espn_scores() -> List[Dict]:
    """
    从ESPN公开API抓取最新比分（无需API Key）
    """
    # ESPN有公开的soccer API，但结构会变化
    # 这里提供框架，实际使用时需调整
    results = []
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                for event in data.get('events', []):
                    try:
                        competition = event['competitions'][0]
                        home = competition['competitors'][0]
                        away = competition['competitors'][1]
                        # 归一化队名
                        home_team_name = normalize_team_name(home['team'].get('displayName', ''))
                        away_team_name = normalize_team_name(away['team'].get('displayName', ''))
                        # ESPN的home/away可能与标准不同，需要按homeAway字段判断
                        if home.get('homeAway') == 'away':
                            home_team_name, away_team_name = away_team_name, home_team_name
                            home_score_val = away.get('score')
                            away_score_val = home.get('score')
                        else:
                            home_score_val = home.get('score')
                            away_score_val = away.get('score')
                        status = event.get('status', {}).get('type', {}).get('name', '')
                        # 状态映射
                        status_short = 'scheduled'
                        if status in ('STATUS_FINAL', 'STATUS_FULL_TIME'):
                            status_short = 'finished'
                        elif status in ('STATUS_IN_PROGRESS', 'STATUS_HALFTIME'):
                            status_short = 'live'
                        elif status == 'STATUS_SCHEDULED':
                            status_short = 'scheduled'
                        # 比分：只在已结束或进行中才有
                        if status_short == 'scheduled':
                            home_score_final = None
                            away_score_final = None
                        else:
                            home_score_final = int(home_score_val) if home_score_val is not None else 0
                            away_score_final = int(away_score_val) if away_score_val is not None else 0
                        results.append({
                            'date': event.get('date', ''),
                            'home_team': home_team_name,
                            'away_team': away_team_name,
                            'home_score': home_score_final,
                            'away_score': away_score_final,
                            'status': status_short,
                            'tournament': 'FIFA World Cup',
                            'neutral': True,
                            'venue': competition.get('venue', {}).get('fullName', ''),
                            'source': 'espn',
                        })
                    except (KeyError, ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"[ESPN] 抓取失败: {e}")
    return results


async def fetch_latest_results() -> List[Dict]:
    """
    主入口：从多个数据源抓取最新比赛
    """
    print("[抓取] 开始获取最新数据...")
    all_results = []

    # 并行抓取多个源
    tasks = [
        fetch_espn_scores(),
        # fetch_wikipedia_world_cup_results(2026),  # 暂时禁用，结构不稳定
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, list):
            all_results.extend(r)
        elif isinstance(r, Exception):
            print(f"[抓取] 源失败: {r}")

    # 去重（按 home+away+date）
    seen = set()
    unique = []
    for r in all_results:
        key = (r.get('home_team', ''), r.get('away_team', ''), str(r.get('date', ''))[:10])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # 缓存
    if unique:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'fetch_time': datetime.now().isoformat(),
                'count': len(unique),
                'results': unique,
            }, f, ensure_ascii=False, indent=2)
        print(f"[抓取] 缓存 {len(unique)} 条到 {CACHE_FILE}")

    return unique


def load_cached_live_data() -> Optional[Dict]:
    """加载缓存的实时数据"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def merge_with_local_results(live_data: List[Dict]) -> dict:
    """
    将抓取的数据合并到本地results.csv
    - 已存在但比分为空：更新比分
    - 不存在：添加新行
    返回：{updated: int, added: int, total: int}
    """
    import pandas as pd
    local_path = DATA_DIR / 'results.csv'
    df = pd.read_csv(local_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    updated_count = 0
    added_count = 0
    need_write = False

    for item in live_data:
        if item.get('home_score') is None or item.get('away_score') is None:
            continue

        try:
            item_date = pd.to_datetime(item['date'], utc=True).date()
        except Exception:
            continue

        mask = (
            (df['home_team'] == item['home_team']) &
            (df['away_team'] == item['away_team']) &
            (df['date'].dt.date == item_date)
        )
        if mask.any():
            idx = df[mask].index[0]
            current_hs = df.at[idx, 'home_score']
            current_as = df.at[idx, 'away_score']
            # 检查是否需要更新（当前为空）
            if pd.isna(current_hs) or pd.isna(current_as):
                df.at[idx, 'home_score'] = int(item['home_score'])
                df.at[idx, 'away_score'] = int(item['away_score'])
                updated_count += 1
                need_write = True
                print(f"[合并] 更新比分: {item['home_team']} {item['home_score']}-{item['away_score']} {item['away_team']}")
        else:
            # 添加新比赛
            new_row = {
                'date': pd.to_datetime(item['date']),
                'home_team': item['home_team'],
                'away_team': item['away_team'],
                'home_score': int(item['home_score']),
                'away_score': int(item['away_score']),
                'tournament': item.get('tournament', 'FIFA World Cup'),
                'city': item.get('city', ''),
                'country': item.get('country', ''),
                'neutral': item.get('neutral', True),
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df['date'] = pd.to_datetime(df['date'])  # 保持 datetime 类型
            added_count += 1
            need_write = True
            print(f"[合并] 新增比赛: {item['home_team']} {item['home_score']}-{item['away_score']} {item['away_team']}")

    if need_write:
        df = df.sort_values('date').reset_index(drop=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_csv(local_path, index=False)

    result = {'updated': updated_count, 'added': added_count, 'total': updated_count + added_count}
    print(f"[合并] 更新 {updated_count} 场, 新增 {added_count} 场")
    return result


if __name__ == "__main__":
    # 测试抓取
    async def test():
        results = await fetch_latest_results()
        print(f"\n获取到 {len(results)} 条数据")
        for r in results[:5]:
            print(f"  {r.get('home_team', '?')} vs {r.get('away_team', '?')}: {r.get('home_score', '?')}-{r.get('away_score', '?')}")
    asyncio.run(test())
