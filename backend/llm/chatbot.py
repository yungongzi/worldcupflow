"""
LLM 对话核心逻辑
- 系统提示词构建
- SSE 流式响应
---
联网搜索已移至 search.py（多层回退：DDGS → Bing → DDG Lite）
"""
import json
from typing import Optional, AsyncGenerator

import httpx

from .config import load_config


# ====================== 系统提示词 ======================

SYSTEM_PROMPT_BASE = """你是一个专业的世界杯足球比赛AI分析助手，名字叫「AI问赛」。

你的职责：
1. 分析足球比赛结果、战术、球员表现
2. 基于历史数据和实时数据提供比赛预测分析
3. 回答用户关于世界杯、球队、球员的问题
4. 解读比赛数据和统计数据

你必须遵守的规则：
- 只回答与足球比赛、世界杯、球队相关的问题
- 如果用户问与足球无关的问题，礼貌拒绝并引导到足球话题
- 回答要专业、客观，引用数据支撑
- 可以使用中文回答，球队名优先使用中文译名
- 预测性内容务必加上「仅供参考」的免责说明"""


def build_system_prompt(match_context: Optional[dict] = None) -> str:
    """构建带比赛上下文的系统提示词"""
    prompt = SYSTEM_PROMPT_BASE
    if not match_context:
        return prompt

    # 添加比赛信息
    home_zh = match_context.get('home_team_zh', match_context.get('home_team', 'N/A'))
    home_en = match_context.get('home_team', '')
    away_zh = match_context.get('away_team_zh', match_context.get('away_team', 'N/A'))
    away_en = match_context.get('away_team', '')

    prompt += f"""

当前比赛信息：
- 主队: {home_zh} ({home_en})
- 客队: {away_zh} ({away_en})
- 状态: {match_context.get('status', 'N/A')}
- 比分: {match_context.get('home_score', '?')} - {match_context.get('away_score', '?')}
- 日期: {match_context.get('date', 'N/A')}
- 场地: {match_context.get('venue', 'N/A')}"""

    # 添加AI预测数据
    if match_context.get('prediction'):
        p = match_context['prediction']
        prompt += f"""
AI预测数据：
- 主胜概率: {p.get('home_win', 0) * 100:.1f}%
- 平局概率: {p.get('draw', 0) * 100:.1f}%
- 客胜概率: {p.get('away_win', 0) * 100:.1f}%
- 预测比分: {p.get('predicted_score', 'N/A')}
- Elo差: {p.get('elo_diff', 'N/A')}"""

    return prompt


# 联网搜索已移至 search.py（DDGS → Bing → DDG Lite 多层回退）

def _sse_event(data: dict) -> str:
    """构建单行 SSE 事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ====================== 流式响应 ======================

async def stream_chat(
    messages: list,
    thinking: bool = False,
) -> AsyncGenerator[str, None]:
    """调用 LLM 并逐 token 输出 SSE 事件

    Yields:
        SSE 格式字符串: "data: {json}\\n\\n"
        其中 json.type 为 "text" | "error" | "done"
    """
    config = load_config()
    api_key = config.get('api_key', '').strip()
    base_url = config.get('base_url', 'https://api.openai.com/v1').rstrip('/')
    model = config.get('model', 'gpt-4o-mini')

    if not api_key:
        yield _sse_event({
            'error': '请先在右上角配置LLM API Key',
            'type': 'error',
        })
        return

    # 思考模式：在 system prompt 追加指令
    if thinking:
        thinking_instruction = (
            "\n\n请对每个问题先进行<thinking>思考分析</thinking>，"
            "然后再给出<answer>正式回答</answer>。"
            "思考过程要展示你的推理链：分析问题→检索相关知识→推理→得出结论。"
        )
        for msg in messages:
            if msg['role'] == 'system':
                msg['content'] = msg['content'] + thinking_instruction

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                'POST',
                f"{base_url}/chat/completions",
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model,
                    'messages': messages,
                    'stream': True,
                    'temperature': 0.7,
                    'max_tokens': 2048,
                },
            ) as response:
                if response.status_code != 200:
                    error_bytes = await response.aread()
                    error_text = error_bytes.decode('utf-8', errors='replace')
                    yield _sse_event({
                        'error': f'LLM API 错误 (HTTP {response.status_code})',
                        'detail': error_text[:500],
                        'type': 'error',
                    })
                    return

                full_content = ""
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_content += content
                                yield _sse_event({
                                    'content': content,
                                    'type': 'text',
                                })
                        except json.JSONDecodeError:
                            continue

                yield _sse_event({
                    'type': 'done',
                    'full': full_content,
                })

    except httpx.ConnectError:
        yield _sse_event({
            'error': f'无法连接到 {base_url}，请检查基础 URL 和网络',
            'type': 'error',
        })
    except Exception as e:
        yield _sse_event({
            'error': f'LLM 调用异常: {str(e)}',
            'type': 'error',
        })
