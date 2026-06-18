"""
LLM 配置管理模块
负责配置的加载、保存、验证
---
★ 安全策略：API Key 从 .env 文件读取，永不回传给前端，永不入库
"""
import json
import os
from pathlib import Path
from pydantic import BaseModel

# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent / 'data' / 'llm_config.json'

# .env 文件路径（项目根目录）
ENV_FILE = Path(__file__).parent.parent.parent / '.env'

# 厂商预设
PROVIDER_PRESETS = {
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
    },
    'deepseek': {
        'base_url': 'https://api.deepseek.com/v1',
        'model': 'deepseek-chat',
    },
    'qwen': {
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen-plus',
    },
    'zhipu': {
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'model': 'glm-4-flash',
    },
    'moonshot': {
        'base_url': 'https://api.moonshot.cn/v1',
        'model': 'moonshot-v1-8k',
    },
}


class LLMConfig(BaseModel):
    """LLM 配置模型"""
    provider: str = 'openai'
    base_url: str = 'https://api.openai.com/v1'
    api_key: str = ''
    model: str = 'gpt-4o-mini'


def _read_env_api_key() -> str:
    """从 .env 文件读取 API Key"""
    env_path = ENV_FILE
    if not env_path.exists():
        return ''
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('LLM_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ''


def load_config() -> dict:
    """
    加载 LLM 配置
    优先级：.env 的 LLM_API_KEY > 配置文件中的 api_key（向后兼容）
    """
    config = PROVIDER_PRESETS['openai'] | {'provider': 'openai', 'api_key': ''}

    # 从配置文件读取 provider/base_url/model
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            config.update(file_config)
        except (json.JSONDecodeError, IOError):
            pass

    # ★ api_key 优先从 .env 读取
    env_key = _read_env_api_key()
    if env_key:
        config['api_key'] = env_key
    # 如果 .env 没有 key，config file 里的 key 也照用（向后兼容）

    return config


def save_config(config: dict) -> None:
    """
    保存 LLM 配置到文件
    ★ 永远不会将 api_key 写入磁盘配置文件
    """
    # 剔除敏感字段
    safe = {k: v for k, v in config.items() if k != 'api_key'}
    # 保留 provider/base_url/model
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)

    # ★ 如果传了新 api_key，同步写入 .env
    new_key = config.get('api_key', '').strip()
    if new_key:
        _write_env_api_key(new_key)


def _write_env_api_key(key: str) -> None:
    """将 API Key 写入 .env 文件"""
    env_path = ENV_FILE
    new_line = f'LLM_API_KEY={key}'

    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            lines = []

        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith('LLM_API_KEY='):
                lines[i] = new_line + '\n'
                found = True
                break

        if not found:
            lines.append(new_line + '\n')

        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    else:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(f'# LLM API Key\n{new_line}\n')


def config_to_frontend(config: dict) -> dict:
    """
    返回给前端的配置
    ★★★ 绝不返回 api_key 明文 ★★★
    前端通过 api_key_masked 判断是否已配置，需要修改时重新输入
    """
    key = config.get('api_key', '')
    masked = key[:4] + '****' + key[-4:] if len(key) > 8 else ('****' if key else '')

    # 只返回安全字段
    return {
        'provider': config.get('provider', 'openai'),
        'base_url': config.get('base_url', ''),
        'model': config.get('model', ''),
        # ★ api_key 字段始终为空字符串（前端用 api_key_masked 判断状态）
        'api_key': '',
        'api_key_masked': masked,
    }


def has_api_key() -> bool:
    """检查是否已配置 API Key"""
    config = load_config()
    return bool(config.get('api_key', '').strip())
