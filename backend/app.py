"""
世界杯预测 - FastAPI 后端
提供预测、球队查询、比赛列表、AI聊天等 API
---
业务逻辑已拆分到:
  - model/        预测模型
  - llm/          AI对话（配置、聊天、搜索）
  - scraping/     实时数据抓取
  - translations/ 中英文队名
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# 添加路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model.predictor import WorldCupPredictor
from model.version_manager import ModelVersionManager
from translations.teams_zh import get_chinese_name, TEAM_ZH
from llm import (
    LLMConfig, load_config, save_config, config_to_frontend, has_api_key,
    build_system_prompt, web_search, stream_chat,
    extract_search_keywords, clean_search_query,
)

# ====================== 应用初始化 ======================

# 全局预测器（启动时加载）
predictor: Optional[WorldCupPredictor] = None

# 模型版本管理器
MODEL_DIR = Path(__file__).parent / 'model' / 'saved'
version_manager = ModelVersionManager(MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型（需要 30-60 秒，遍历 49,425 场历史比赛初始化 Elo）"""
    global predictor
    print("[启动] 开始加载预测器（预计 30-60 秒，请勿在加载完成前访问接口）...")
    print("[启动] 步骤 1/3: 加载 XGBoost 模型文件...")
    predictor = WorldCupPredictor()
    print("[启动] 步骤 2/3: 模型加载完成")
    print("[启动] 步骤 3/3: 服务就绪，可以访问 http://localhost:8018")
    yield
    # shutdown 阶段（当前无需清理资源）
    print("[关闭] 服务正在停止...")


app = FastAPI(
    title="世界杯预测 API",
    description="基于 XGBoost + Poisson 的国际足球比赛预测系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== 请求/响应模型 ======================

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str = "FIFA World Cup"
    neutral: bool = True


class ChatRequest(BaseModel):
    message: str
    thinking: bool = False
    web_search: bool = False
    match_context: Optional[dict] = None


# ====================== 兜底静态文件路由 ======================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """提供网站图标"""
    favicon_path = Path(__file__).parent.parent / 'frontend' / 'favicon.ico'
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type='image/x-icon')
    return FileResponse(Path(__file__).parent.parent / 'frontend' / 'index.html')

@app.get("/")
async def root():
    frontend_path = Path(__file__).parent.parent / 'frontend' / 'index.html'
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "WorldCup Predictor API", "docs": "/docs"}


@app.get("/style.css")
async def style_css():
    p = Path(__file__).parent.parent / 'frontend' / 'style.css'
    if p.exists():
        return FileResponse(p, media_type='text/css')
    raise HTTPException(404)


@app.get("/script.js")
async def script_js():
    p = Path(__file__).parent.parent / 'frontend' / 'script.js'
    if p.exists():
        return FileResponse(p, media_type='application/javascript')
    raise HTTPException(404)


# ====================== 核心预测 API ======================

@app.get("/api/health")
async def health():
    active_version = version_manager.get_active_version_id()
    loaded_version = predictor.get_current_version() if predictor else None
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "active_version": active_version,
        "loaded_version": loaded_version,
        "training_metadata": predictor.metadata if predictor else None,
    }


@app.get("/api/teams")
async def list_teams(top_n: int = Query(80, ge=1, le=500)):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    return predictor.get_team_list(top_n)


@app.get("/api/teams/search")
async def search_teams(q: str = Query(..., min_length=1)):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    return predictor.search_team(q)


@app.post("/api/predict")
async def predict_match(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    try:
        return predictor.predict(req.home_team, req.away_team, req.tournament, req.neutral)
    except Exception as e:
        raise HTTPException(500, f"预测失败: {str(e)}")


@app.get("/api/predict")
async def predict_match_get(
    home: str = Query(...),
    away: str = Query(...),
    tournament: str = Query("FIFA World Cup"),
    neutral: bool = Query(True),
):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    try:
        return predictor.predict(home, away, tournament, neutral)
    except Exception as e:
        raise HTTPException(500, f"预测失败: {str(e)}")


@app.get("/api/worldcup/schedule")
async def worldcup_schedule(limit: int = Query(50, ge=1, le=200)):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    matches = predictor.get_upcoming_world_cup_matches(limit)
    return {"count": len(matches), "matches": matches}


@app.get("/api/worldcup/full-schedule")
async def full_schedule():
    """返回 2026 世界杯完整赛程（小组赛 + 淘汰赛），含比分和轮次信息"""
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    matches = predictor.get_full_schedule()
    # 统计各轮次
    phase_counts = {}
    for m in matches:
        phase_counts[m['phase']] = phase_counts.get(m['phase'], 0) + 1
    return {
        "count": len(matches),
        "phase_counts": phase_counts,
        "matches": matches,
    }


@app.post("/api/worldcup/update-scores")
async def update_worldcup_scores():
    """从 ESPN API 抓取已完赛世界杯比赛比分，自动更新 results.csv"""
    from scraping.scraper import fetch_latest_results, merge_with_local_results
    try:
        live_data = await fetch_latest_results()
        if not live_data:
            return {"status": "ok", "message": "未获取到新数据", "updated": 0, "added": 0}
        result = merge_with_local_results(live_data)
        return {
            "status": "ok",
            "message": f"更新 {result['updated']} 场比分, 新增 {result['added']} 场比赛",
            **result,
        }
    except Exception as e:
        raise HTTPException(500, f"更新比分失败: {str(e)}")


@app.post("/api/worldcup/manual-score")
async def manual_update_score(
    home_team: str = Query(...),
    away_team: str = Query(...),
    match_date: str = Query(...),
    home_score: int = Query(...),
    away_score: int = Query(...),
):
    """手动更新单场比赛比分"""
    import pandas as pd
    DATA_DIR = Path(__file__).parent / 'data'
    csv_path = DATA_DIR / 'results.csv'
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        mask = (
            (df['home_team'] == home_team) &
            (df['away_team'] == away_team) &
            (df['date'].dt.date == pd.to_datetime(match_date).date())
        )
        if not mask.any():
            raise HTTPException(404, f"未找到比赛: {home_team} vs {away_team} ({match_date})")
        idx = df[mask].index[0]
        df.at[idx, 'home_score'] = home_score
        df.at[idx, 'away_score'] = away_score
        df.to_csv(csv_path, index=False)
        return {
            "status": "ok",
            "message": f"已更新: {home_team} {home_score}-{away_score} {away_team}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"更新失败: {str(e)}")


@app.get("/api/results/recent")
async def recent_results(limit: int = Query(20, ge=1, le=100)):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    return predictor.get_recent_results(limit)


@app.get("/api/worldcup/predict-all")
async def predict_all_worldcup(limit: int = Query(20, ge=1, le=100)):
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    matches = predictor.get_upcoming_world_cup_matches(limit)
    predictions = []
    for m in matches:
        try:
            pred = predictor.predict(m['home_team'], m['away_team'], 'FIFA World Cup', True)
            predictions.append({
                'date': m['date'],
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'home_team_zh': m['home_team_zh'],
                'away_team_zh': m['away_team_zh'],
                'city': m['city'],
                'country': m['country'],
                'prediction': {
                    'home_win': pred['probabilities']['home_win'],
                    'draw': pred['probabilities']['draw'],
                    'away_win': pred['probabilities']['away_win'],
                    'predicted_score': f"{pred['predicted_score']['home']}-{pred['predicted_score']['away']}",
                    'top_score': pred['top_scores'][0]['score'],
                    'top_score_prob': pred['top_scores'][0]['probability'],
                },
            })
        except Exception as e:
            print(f"预测 {m['home_team']} vs {m['away_team']} 失败: {e}")
    return {"count": len(predictions), "predictions": predictions}


@app.get("/api/live")
async def live_matches():
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")
    from scraping.scraper import fetch_latest_results, load_cached_live_data
    try:
        live_data = await fetch_latest_results()
        if not live_data:
            cached = load_cached_live_data()
            live_data = cached.get('results', []) if cached else []
    except Exception as e:
        print(f"实时抓取失败，使用缓存: {e}")
        cached = load_cached_live_data()
        live_data = cached.get('results', []) if cached else []

    enriched = []
    for m in live_data:
        item = {
            **m,
            'home_team_zh': get_chinese_name(m['home_team']),
            'away_team_zh': get_chinese_name(m['away_team']),
            'prediction': None,
        }
        if m.get('status') in ('scheduled', 'live') or m.get('home_score') is None:
            try:
                pred = predictor.predict(m['home_team'], m['away_team'], 'FIFA World Cup', True)
                item['prediction'] = {
                    'home_win': pred['probabilities']['home_win'],
                    'draw': pred['probabilities']['draw'],
                    'away_win': pred['probabilities']['away_win'],
                    'predicted_score': f"{pred['predicted_score']['home']}-{pred['predicted_score']['away']}",
                    'top_scores': pred['top_scores'][:3],
                    'explanation': pred['explanation'][:2],
                    'elo_diff': pred['elo_ratings']['diff'],
                }
            except Exception as e:
                print(f"预测 {m['home_team']} vs {m['away_team']} 失败: {e}")
        enriched.append(item)
    return {"count": len(enriched), "fetch_time": datetime.now().isoformat(), "matches": enriched}


@app.get("/api/tournaments")
async def list_tournaments():
    from model.features import TOURNAMENT_WEIGHT
    mapping = {
        'FIFA World Cup': '世界杯',
        'FIFA World Cup qualification': '世界杯预选赛',
        'UEFA Euro': '欧洲杯',
        'UEFA Euro qualification': '欧洲杯预选赛',
        'Copa América': '美洲杯',
        'African Cup of Nations': '非洲杯',
        'African Cup of Nations qualification': '非洲杯预选赛',
        'AFC Asian Cup': '亚洲杯',
        'AFC Asian Cup qualification': '亚洲杯预选赛',
        'Gold Cup': '金杯赛',
        'CONCACAF Nations League': '中北美国家联赛',
        'UEFA Nations League': '欧洲国家联赛',
        'Confederations Cup': '联合会杯',
        'Friendly': '友谊赛',
    }
    return [
        {"name": k, "weight": v, "name_zh": mapping.get(k, k)}
        for k, v in sorted(TOURNAMENT_WEIGHT.items(), key=lambda x: -x[1])
    ]


@app.post("/api/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(_refresh_data_task)
    return {"message": "数据更新任务已启动", "status": "processing"}


async def _refresh_data_task():
    try:
        print("[刷新] 开始数据更新...")
        from scraping.scraper import fetch_latest_results
        new_data = await fetch_latest_results()
        if new_data:
            print(f"[刷新] 获取到 {len(new_data)} 条新数据")
        print("[刷新] 数据更新完成")
    except Exception as e:
        print(f"[刷新] 失败: {e}")


# ====================== 模型版本管理 API ======================

@app.get("/api/models")
async def list_models():
    """列出所有模型版本"""
    versions = version_manager.list_versions()
    active = version_manager.get_active_version_id()
    return {
        "active_version": active,
        "loaded_version": predictor.get_current_version() if predictor else None,
        "versions": versions,
        "count": len(versions),
    }


@app.post("/api/models/{version_id}/activate")
async def activate_model(version_id: str):
    """切换活跃版本（同步文件到根目录并重新加载预测器）"""
    global predictor
    try:
        ok = version_manager.activate_version(version_id)
        if not ok:
            raise HTTPException(404, f"版本不存在: {version_id}")
        # 重新加载预测器（从根目录加载活跃版本）
        predictor.reload()
        return {
            "status": "ok",
            "active_version": version_id,
            "message": f"已切换活跃版本到 {version_id}",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"切换版本失败: {str(e)}")


@app.delete("/api/models/{version_id}")
async def delete_model(version_id: str):
    """删除指定版本（不能删除活跃版本）"""
    try:
        ok = version_manager.delete_version(version_id)
        if not ok:
            raise HTTPException(404, f"版本不存在: {version_id}")
        return {"status": "ok", "message": f"已删除版本 {version_id}"}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"删除版本失败: {str(e)}")


# ====================== LLM 配置 API ======================

@app.get("/api/config/llm")
async def get_llm_config():
    """获取 LLM 配置（含部分隐藏的 api_key）"""
    config = load_config()
    return config_to_frontend(config)


@app.post("/api/config/llm")
async def save_llm_config(config: LLMConfig):
    """保存 LLM 配置"""
    save_config(config.model_dump())
    return {"status": "ok", "message": "配置已保存"}


# ====================== AI 聊天 API ======================

@app.post("/api/chat")
async def chat_with_ai(req: ChatRequest):
    """AI 聊天 — SSE 流式响应"""
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")

    # 构建消息
    system_prompt = build_system_prompt(req.match_context)
    messages = [{"role": "system", "content": system_prompt}]

    # 联网搜索（注入搜索结果）
    if req.web_search:
        # ★ 清洗查询：从消息中提取有效搜索关键词（去除换行等非法字符）
        search_query = clean_search_query(req.message, max_len=200)
        if not search_query:
            search_query = extract_search_keywords(req.message)

        print(f"[搜索请求] 原始长度={len(req.message)} 清洗后={search_query[:80]}...")
        search_results = await web_search(search_query)
        if search_results:
            messages.append({
                "role": "system",
                "content": (
                    "以下是最新的网络搜索结果（请综合这些信息回答，并在回答中引用关键数据来源）：\n"
                    f"{search_results}"
                ),
            })
        else:
            # 搜索失败 — 通知 LLM 无搜索结果可用
            messages.append({
                "role": "system",
                "content": (
                    "注意：本次启用了联网搜索，但未能获取到有效的搜索结果。"
                    "请基于你的训练数据回答问题，并在开头提示用户「当前联网搜索暂不可用，以下回答基于模型训练数据」。"
                ),
            })

    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        stream_chat(messages, req.thinking),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chat/match-analysis")
async def match_analysis(
    home: str = Query(...),
    away: str = Query(...),
    status: str = Query("scheduled"),
    home_score: Optional[str] = Query(None),
    away_score: Optional[str] = Query(None),
):
    """生成比赛分析提示词（供 AI问赛 按钮使用）"""
    if predictor is None:
        raise HTTPException(503, "模型尚未加载完成")

    home_zh = get_chinese_name(home)
    away_zh = get_chinese_name(away)

    # ★ 实际比分
    has_actual_score = home_score is not None and away_score is not None and home_score != '' and away_score != ''
    actual_score_line = f"  **实际比分：{home_zh} {home_score} - {away_score} {away_zh}**" if has_actual_score else ""

    # AI 模型预测
    prediction = None
    try:
        pred = predictor.predict(home, away, 'FIFA World Cup', True)
        prediction = {
            'home_win': pred['probabilities']['home_win'],
            'draw': pred['probabilities']['draw'],
            'away_win': pred['probabilities']['away_win'],
            'predicted_score': f"{pred['predicted_score']['home']}-{pred['predicted_score']['away']}",
            'elo_diff': pred['elo_ratings'].get('diff', 0),
        }
    except Exception:
        pass

    # 历史交锋
    h2h_text = ""
    try:
        h2h = predictor.get_h2h_records(home, away, limit=5)
        if h2h:
            h2h_lines = [
                f"  {m['date'][:10]} {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}"
                for m in h2h
            ]
            h2h_text = "\n".join(h2h_lines)
    except Exception:
        pass

    pred_json = json.dumps(prediction, ensure_ascii=False, indent=2) if prediction else '暂无预测数据'

    if status in ('finished', 'live'):
        question = f"""请分析 {home_zh}({home}) vs {away_zh}({away}) 这场比赛的过程和原因。

实际比赛信息：
{actual_score_line if actual_score_line else '暂无实际比分数据'}

历史交锋记录（近5场）：
{h2h_text if h2h_text else '暂无数据'}

AI模型预测数据：
{pred_json}

请分析：
1. 比赛过程和关键节点（参考实际比分）
2. 胜负原因分析（战术、数据支撑）
3. 关键球员表现评估
4. 对本届世界杯的影响"""
    else:
        question = f"""请分析并预测 {home_zh}({home}) vs {away_zh}({away}) 这场即将到来的世界杯比赛。

历史交锋记录（近5场）：
{h2h_text if h2h_text else '暂无数据'}

AI模型预测数据：
{pred_json}

请分析：
1. 两队实力对比（近期状态、历史交锋）
2. 可能的战术布局
3. 预测比赛走势和关键因素
4. 比分预测（请说明预测依据）
注意：预测仅供参考，足球比赛存在不确定性。"""

    return {
        "question": question,
        "home_team_zh": home_zh,
        "away_team_zh": away_zh,
        "status": status,
        "prediction": prediction if has_actual_score else None,
    }


# ====================== 静态文件服务 ======================

frontend_dir = Path(__file__).parent.parent / 'frontend'
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("  世界杯预测系统启动")
    print("  访问地址: http://localhost:8018")
    print("  API文档:   http://localhost:8018/docs")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8018, log_level="info")
