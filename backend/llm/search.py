"""
联网搜索模块 — 多层回退 + 自动清洗 + 异步安全
===============================================
回退链路: DDGS(Google/Bing) → Bing直接抓取 → DDG Lite → 空结果
"""
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx

# 线程池用于包装同步 ddgs 调用（避免阻塞事件循环）
_search_executor = ThreadPoolExecutor(max_workers=2)


# ====================== 查询清洗 ======================

def clean_search_query(query: str, max_len: int = 200) -> str:
    """清洗搜索查询：
    - 去除换行 / 回车 / 多余空白
    - 截断过长的文本（避免 URL 超长和无效匹配）
    - 保留中文、英文、数字和常用标点
    """
    # 替换所有换行为空格
    q = query.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    # 压缩连续空格
    q = re.sub(r'\s{2,}', ' ', q).strip()
    # 截断
    if len(q) > max_len:
        # 优先在空格处截断
        cut = q[:max_len].rfind(' ')
        if cut > max_len // 2:
            q = q[:cut]
        else:
            q = q[:max_len]
    return q


def extract_search_keywords(message: str) -> str:
    """从完整对话消息中提取可用于搜索的关键词

    对于 match-analysis 多行问题，只取第一行 + 关键词
    对于普通消息，直接用 clean_search_query
    """
    cleaned = clean_search_query(message, max_len=300)

    # 如果消息很短，直接用
    if len(cleaned) <= 80:
        return cleaned

    # 提取标题行（"请分析 XXX vs XXX" 这种模式）
    title_match = re.search(r'(请分析|请预测|分析|预测)\s*[：:]*\s*(.+?)(?:\n|。|$)', cleaned)
    if title_match:
        core = title_match.group(2).strip()
        # 加上世界杯前缀
        return f"世界杯 {core}"

    # 取前 150 字
    return cleaned[:150]


def _build_search_query(user_message: str, force_prefix: bool = True) -> str:
    """构建最终的搜索查询字符串"""
    cleaned = extract_search_keywords(user_message)

    # 自动补充"世界杯"前缀（足球场景专属）
    if force_prefix and '世界杯' not in cleaned and 'World Cup' not in cleaned.lower():
        cleaned = f"世界杯 {cleaned}"

    return clean_search_query(cleaned, max_len=200)


# ====================== 搜索实现 ======================

def _search_ddgs_sync(query: str, max_results: int) -> list[dict]:
    """通过 ddgs 库搜索（同步，在 executor 中运行）"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
                for r in raw
            ]
    except ImportError:
        raise RuntimeError("ddgs 库未安装，请运行: pip install ddgs")
    except Exception as e:
        raise RuntimeError(f"DDGS 搜索异常: {e}")


async def _search_with_ddgs(query: str, max_results: int) -> list[dict]:
    """异步包装 ddgs 搜索"""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_search_executor, _search_ddgs_sync, query, max_results),
            timeout=12.0,
        )
    except asyncio.TimeoutError:
        print("[搜索] DDGS 超时")
        return []
    except Exception as e:
        print(f"[搜索] DDGS 失败: {e}")
        return []


async def _search_with_bing_direct(query: str, max_results: int) -> list[dict]:
    """直接抓取 Bing 搜索结果页（正确 URL 编码）"""
    query_enc = quote(query, safe='')
    url = f"https://www.bing.com/search?q={query_enc}&setlang=zh-cn"

    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
            if resp.status_code != 200:
                print(f"[搜索] Bing 返回 HTTP {resp.status_code}")
                return []

            html = resp.text

            # 提取搜索结果项
            results: list[dict] = []
            # Bing 的搜索结果在 <li class="b_algo"> 中
            items = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', html, re.DOTALL)
            if not items:
                # 尝试新版标记
                items = re.findall(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>', html, re.DOTALL)

            for item in items[:max_results]:
                # 提取标题
                title_m = re.search(r'<h2[^>]*>.*?<a[^>]*>(.*?)</a>', item, re.DOTALL)
                # 提取链接
                href_m = re.search(r'<a[^>]*href="(https?://[^"]+)"', item)
                # 提取摘要
                snippet_m = re.search(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', item, re.DOTALL)
                if not snippet_m:
                    snippet_m = re.search(r'<p[^>]*>(.*?)</p>', item, re.DOTALL)

                title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""
                href = href_m.group(1) if href_m else ""
                body = re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip() if snippet_m else ""

                if title and href and len(body) > 15:
                    results.append({"title": title, "href": href, "body": body})

            return results

    except Exception as e:
        print(f"[搜索] Bing 直接抓取失败: {e}")
        return []


async def _search_with_ddg_lite(query: str, max_results: int) -> list[dict]:
    """抓取 DuckDuckGo Lite 版（HTML 较简单，反爬较弱）"""
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            if resp.status_code != 200:
                print(f"[搜索] DDG Lite 返回 HTTP {resp.status_code}")
                return []

            html = resp.text
            results: list[dict] = []

            # DDG Lite 的结果格式：<a rel="nofollow" href="...">title</a><span class="link-text">url</span><span class="snippet">...</span>
            # 或者更简单的 <tr> 行格式
            rows = re.findall(r'<tr[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL)
            if not rows:
                rows = re.findall(r'<a rel="nofollow" href="(https?://[^"]+)">(.*?)</a>.*?class="[^"]*snippet[^"]*"[^>]*>(.*?)</', html, re.DOTALL)
                for href, title, snippet in rows[:max_results]:
                    clean_title = re.sub(r'<[^>]+>', '', title).strip()
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    if clean_title and clean_snippet:
                        results.append({"title": clean_title, "href": href, "body": clean_snippet})
                return results

            for row in rows[:max_results]:
                href_m = re.search(r'href="(https?://[^"]+)"', row)
                title_m = re.search(r'<a[^>]*>(.*?)</a>', row)
                snippet_m = re.search(r'class="[^"]*snippet[^"]*"[^>]*>(.*?)</', row, re.DOTALL)
                if not snippet_m:
                    snippet_m = re.search(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)

                href = href_m.group(1) if href_m else ""
                title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""
                snippet = re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip() if snippet_m else ""

                if title and snippet:
                    results.append({"title": title, "href": href, "body": snippet})

            return results

    except Exception as e:
        print(f"[搜索] DDG Lite 失败: {e}")
        return []


# ====================== 对外接口 ======================

def format_search_results(results: list[dict]) -> str:
    """将搜索结果列表格式化为 LLM 可读文本"""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"[{i + 1}] {title}")
        lines.append(f"    {body}")
        if href:
            lines.append(f"    来源: {href}")
    return "\n".join(lines)


async def web_search(query: str, max_results: int = 5, user_message: str = "") -> str:
    """多层回退联网搜索

    Args:
        query: 搜索查询字符串
        max_results: 最大结果数
        user_message: 原始用户消息（用于提取更好的搜索关键词）

    Returns:
        格式化的搜索结果文本，失败时返回空字符串
    """
    # 清洗查询（去除换行等非法字符）
    clean_q = clean_search_query(query, max_len=200)

    if not clean_q:
        print("[搜索] 查询为空，跳过搜索")
        return ""

    print(f"[搜索] 查询: {clean_q[:80]}...")

    all_results: list[dict] = []
    errors: list[str] = []

    # === Layer 1: DDGS（最可靠） ===
    print("[搜索] 尝试 DDGS...")
    try:
        results = await _search_with_ddgs(clean_q, max_results)
        if results:
            all_results = results
            print(f"[搜索] DDGS 成功，获取 {len(results)} 条结果")
        else:
            errors.append("DDGS 返回空结果")
    except Exception as e:
        errors.append(f"DDGS: {e}")
        print(f"[搜索] DDGS 异常: {e}")

    # === Layer 2: Bing 直接抓取 ===
    if not all_results:
        print("[搜索] 尝试 Bing 直接抓取...")
        try:
            results = await _search_with_bing_direct(clean_q, max_results)
            if results:
                all_results = results
                print(f"[搜索] Bing 成功，获取 {len(results)} 条结果")
            else:
                errors.append("Bing 返回空结果")
        except Exception as e:
            errors.append(f"Bing: {e}")
            print(f"[搜索] Bing 异常: {e}")

    # === Layer 3: DDG Lite ===
    if not all_results:
        print("[搜索] 尝试 DDG Lite...")
        try:
            results = await _search_with_ddg_lite(clean_q, max_results)
            if results:
                all_results = results
                print(f"[搜索] DDG Lite 成功，获取 {len(results)} 条结果")
            else:
                errors.append("DDG Lite 返回空结果")
        except Exception as e:
            errors.append(f"DDG Lite: {e}")
            print(f"[搜索] DDG Lite 异常: {e}")

    # === 返回结果 ===
    if all_results:
        return format_search_results(all_results)

    # 全部失败 — 返回空字符串，由上层模块告知用户
    error_summary = "; ".join(errors) if errors else "未知错误"
    print(f"[搜索] ⚠ 所有搜索源均失败: {error_summary}")
    return ""
