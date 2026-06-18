"""
LLM 对话模块
提供配置管理、AI聊天、联网搜索等能力

对外接口:
  - LLMConfig          配置模型 (Pydantic)
  - load_config()      加载配置
  - save_config()      保存配置
  - config_to_frontend() 前端友好的配置
  - has_api_key()      是否已配置API Key
  - build_system_prompt() 构建系统提示词
  - web_search()       联网搜索（多层回退）
  - stream_chat()      SSE流式对话
"""

from .config import (
    LLMConfig,
    load_config,
    save_config,
    config_to_frontend,
    has_api_key,
    PROVIDER_PRESETS,
)

from .chatbot import (
    build_system_prompt,
    stream_chat,
)

from .search import (
    web_search,
    clean_search_query,
    extract_search_keywords,
)

__all__ = [
    'LLMConfig',
    'load_config',
    'save_config',
    'config_to_frontend',
    'has_api_key',
    'PROVIDER_PRESETS',
    'build_system_prompt',
    'stream_chat',
    'web_search',
    'clean_search_query',
    'extract_search_keywords',
]
