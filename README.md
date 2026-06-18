# ⚽ 世界杯 AI 预测系统 · World Cup AI Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-3b5a2b?style=flat" alt="XGBoost">
  <img src="https://img.shields.io/badge/LLM-多厂商兼容-orange?style=flat" alt="LLM">
  <img src="https://img.shields.io/badge/数据量-49,000%2B_场-blue?style=flat" alt="Dataset">
</p>

> **基于 49,000+ 场国际足球历史比赛数据、XGBoost + Poisson 双模型架构的 AI 世界杯预测系统。**  
> 融合 Elo 等级分、近期状态、历史交锋、赛事权重等 39 维特征，实时抓取 ESPN 数据，可配置 AI 对话助手。

---

⚠️ **重要提示 / Disclaime**

**预测结果仅供参考，足球比赛存在高度不确定性，请理性看待预测结果**

**实时预测基于历史统计模型，未考虑伤病、红牌、伤停补时等临场因素，仅供参考**

---

## 🌟 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **双模型预测** | XGBoost 胜平负分类 + Poisson 进球数回归，输出比分概率矩阵 |
| 📊 **39 维特征** | Elo 等级分、近 5/10 场状态、H2H 交锋、赛事权重、攻防衍生特征 |
| 🌐 **实时数据** | ESPN API 自动抓取世界杯赛程、比分、状态，预测与实况联动 |
| 💬 **AI 对话** | 内置 OpenAI 兼容接口的聊天助手，SSE 流式输出，支持思考模式 |
| 🔍 **联网搜索** | ddgs 库驱动的三级回退搜索引擎（自动提取关键词、清洗查询） |
| 🎨 **玻璃拟态** | 内高光渐变、四向异色边框、saturate(180%)、多层阴影的前端视觉 |

---

## 📁 项目结构

```
worldcup-predictor/
├── .env                       # 🔒 API Key
├── .gitignore                 # 安全排除规则
├── README.md                  # 本文件
│
├── backend/                   # Python 后端（FastAPI）
│   ├── app.py                 #   主应用入口 · 所有 API 路由 · 启动逻辑
│   ├── requirements.txt       #   Python 依赖清单
│   │
│   ├── model/                 # 🧠 预测模型模块
│   │   ├── features.py        #   特征工程：Elo 系统 · 赛事权重 · 状态提取
│   │   ├── train.py           #   模型训练脚本（XGBoost 分类 + Poisson 回归）
│   │   ├── predictor.py       #   预测器类：加载模型 · 预测 · H2H 查询
│   │   └── saved/             #   训练好的模型文件（不入 git）
│   │       ├── outcome_model.json         XGBoost 胜平负分类器
│   │       ├── home_score_model.json      Poisson 主队进球回归
│   │       ├── away_score_model.json      Poisson 客队进球回归
│   │       ├── elo_ratings.json           336 支国家队 Elo 等级分
│   │       ├── feature_columns.json       39 维特征列顺序
│   │       └── training_metadata.json     训练元数据及指标
│   │
│   ├── llm/                   # 💬 AI 对话模块
│   │   ├── __init__.py        #   模块入口 · 统一导出
│   │   ├── config.py          #   配置管理：.env 读取 · 6 厂商预设 · 安全脱敏
│   │   ├── chatbot.py         #   SSE 流式聊天 · 系统提示词构建 · 思考模式
│   │   └── search.py          #   联网搜索：ddgs 主引擎 · Bing · DDG Lite 三级回退
│   │
│   ├── scraping/              # 📡 实时数据抓取
│   │   └── scraper.py         #   ESPN FIFA World Cup API 抓取 + 本地缓存
│   │
│   ├── translations/          # 🌍 国际化
│   │   └── teams_zh.py        #   336 支球队中英文名称映射
│   │
│   └── data/                  # 📦 原始数据
│       ├── results.csv        #   49,425 场国际比赛（1872-2026）
│       ├── goalscorers.csv    #   47,663 条进球记录
│       ├── shootouts.csv      #   677 场点球大战
│       ├── former_names.csv   #   36 支球队历史名称映射
│       └── live_cache.json    #   实时数据本地缓存
│
├── frontend/                  # 🎨 前端（纯 HTML/CSS/JS）
│   ├── index.html             #   主页面 · 玻璃拟态布局 · 三个功能面板
│   ├── style.css              #   样式表 · 2000+ 行 · CSS 变量体系
│   └── script.js              #   逻辑脚本 · 1400+ 行 · rAF 动画 · SSE 流式
│
└── notebooks/                 # 📓 分析笔记
    └── analyze_time_range.py  #   训练时间范围策略验证
```

---

## 🏗️ 技术架构

### 数据策略：混合时间策略

| 组件 | 时间范围 | 数据量 | 设计理由 |
|------|---------|--------|---------|
| Elo 等级分 | 1872—2025 全历史 | 49,425 场 | Elo 需要长期积累才能稳定收敛 |
| ML 训练数据 | 2002—2025 现代足球 | ~19,673 场 | 反映当代战术风格、规则与竞技水平 |
| 测试集 | 2023—2026 | ~3,618 场 | 时间外验证 (Temporal Out-of-Sample)，杜绝数据泄漏 |

**实证验证**：对 8 种不同时间范围训练的简单模型进行对比测试，全量数据 (1872+) 准确率最高 (59.56%)，但与现代数据 (2002+) 差距极小 (59.11%)。混合策略在保持 Elo 稳定性的同时，使用现代数据训练高维特征，兼顾统计鲁棒性和时代代表性。

### 双模型预测流程

```
历史数据 (49,425场)
       │
       ├──→ Elo 评分系统 ──→ 336 队等级分
       │
       └──→ 39 维特征工程 ──→ XGBoost 分类器 ──→ 胜 / 平 / 负概率
                         │
                         └──→ XGBoost Poisson 回归 ──→ 主队进球 λ₁ · 客队进球 λ₂
                                                      │
                                                      └──→ 二维 Poisson 分布 ──→ Top 5 最可能比分
```

### 模型一：XGBoost 胜平负分类器

- **目标**：预测主胜 / 平局 / 客胜三分类概率
- **特征**：39 维（Elo × 5 + 近期状态 × 20 + H2H × 4 + 上下文 × 5 + 衍生 × 5）
- **优化**：`scale_pos_weight` 类别不平衡处理、早停 (early_stopping_rounds) 防止过拟合
- **输出**：三分类概率 + Top 3 预测理由（Shapley 特征贡献度解释）

### 模型二：XGBoost Poisson 回归

- **目标**：分别预测主队和客队的期望进球数 (λ)
- **损失函数**：`objective='count:poisson'`，天然适合计数数据
- **分离训练**：两套独立模型分别拟合主队和客队进球分布
- **比分构建**：P(比分 = i:j) = Poisson(i; λ_home) × Poisson(j; λ_away)，归一化后输出最可能的 5 个比分

### 特征工程：39 维特征全景

| 类别 | 特征名称 | 数量 | 说明 |
|------|---------|:--:|------|
| **Elo 等级分** | home_elo / away_elo / elo_diff / elo_ratio / elo_normalized | 5 | 全历史积累的等级分及衍生比率 |
| **近期状态 (5 场)** | 胜率 / 净胜球 / 进球 / 失球 / 加权积分 × 主客各 1 组 | 10 | 近期 5 场加权 (越近权重越高) |
| **近期状态 (10 场)** | 同上 × 主客各 1 组 | 10 | 中期稳定性指标 |
| **历史交锋 (H2H)** | 近 5 场胜率 / 平局率 / 净胜球 / 总交锋次数 | 4 | 球队间的历史克制关系 |
| **比赛上下文** | 中立场 / 赛事权重 / 间隔天数 / 主客身份 / 排名差 | 5 | 比赛环境与赛程密度 |
| **攻防衍生** | 近期状态差值 / 攻防能力差值 / Elo×赛事权重交互等 | 5 | 特征交互与非线性组合 |


---

## 📡 API 端点总览

| 端点 | 方法 | 说明 |
|------|:----:|------|
| `/` | GET | 前端主页面 |
| `/api/health` | GET | 健康检查 + 模型加载状态 + 训练元数据 |
| `/api/teams` | GET | 球队列表（按 Elo 降序，支持 top_n 分页） |
| `/api/teams/search?q=xxx` | GET | 球队搜索（中/英文模糊匹配） |
| `/api/predict` | POST/GET | 单场比赛预测（指定两队 + 赛事） |
| `/api/live` | GET | 实时比赛数据（ESPN 抓取 + AI 预测联动） |
| `/api/worldcup/schedule` | GET | 世界杯赛程列表 |
| `/api/worldcup/predict-all` | GET | 批量预测所有即将到来的世界杯比赛 |
| `/api/results/recent` | GET | 近期比赛结果 |
| `/api/refresh` | POST | 触发后台数据刷新 |
| `/api/tournaments` | GET | 支持的赛事列表及权重 |
| `/api/chat` | POST | AI 对话（SSE 流式，支持思考模式 + 联网搜索） |
| `/api/chat/match-analysis` | GET | 生成比赛分析提示词（含比分 + H2H + 预测数据） |
| `/api/config/llm` | GET/POST | LLM 配置读取与保存（API Key 安全脱敏） |
| `/docs` | GET | Swagger UI 交互式 API 文档 |

---

## 🚀 部署教程

### 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.11+ | 推荐 3.13（项目已验证） |
| pip | 23.0+ | Python 包管理器 |
| 磁盘空间 | ~15 MB | 模型文件 + 数据 + 依赖包 |
| 内存 | ~500 MB | 启动时加载 49,425 场比赛初始化 Elo |

---

### 第一步：克隆项目 & 进入目录

```bash
cd worldcupflow
```

---

### 第二步：配置 Python 虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 安装依赖
pip install -r backend/requirements.txt
```

---

### 第三步：配置 LLM API Key（必须）

AI 对话功能依赖 OpenAI 兼容接口的 LLM 服务。在项目根目录创建 `.env` 文件：

```bash
# 创建 .env 文件，写入你的 API Key
echo 'LLM_API_KEY=sk-your-api-key-here' > .env
```

| 变量 | 必填 | 说明 |
|------|:--:|------|
| `LLM_API_KEY` | ✅ | LLM 服务的 API Key（OpenAI 兼容格式） |

**支持的 LLM 厂商**（前端配置面板可选，均使用 OpenAI 兼容 API 路径）：

| 厂商 | 默认模型 | Base URL |
|------|---------|----------|
| **OpenAI** | gpt-4o-mini | `https://api.openai.com/v1` |
| **DeepSeek** | deepseek-chat | `https://api.deepseek.com/v1` |
| **通义千问 (Qwen)** | qwen-plus | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| **智谱 (GLM)** | glm-4-flash | `https://open.bigmodel.cn/api/paas/v4` |
| **Moonshot** | moonshot-v1-8k | `https://api.moonshot.cn/v1` |
| **自定义** | 任意 | 任意 OpenAI 兼容地址 |

---

### 第四步：确认模型文件存在

确保 `backend/model/saved/` 目录下有以下文件（默认已包含训练好的模型）：

```
backend/model/saved/
├── outcome_model.json
├── home_score_model.json
├── away_score_model.json
├── elo_ratings.json
├── feature_columns.json
└── training_metadata.json
```

> ⚠️ 如果初次克隆项目缺少这些文件，需要先运行训练脚本生成：`cd backend && python model/train.py`

---

### 第五步：启动服务

```bash
cd backend
python app.py
```

启动后输出：

```
======================================================================
  世界杯预测系统启动
  访问地址: http://localhost:8018
  API文档:   http://localhost:8018/docs
======================================================================
[启动] 开始加载预测器（预计 30-60 秒，请勿在加载完成前访问接口）...
[启动] 步骤 1/3: 加载 XGBoost 模型文件...
[启动] 步骤 2/3: 模型加载完成
[启动] 步骤 3/3: 服务就绪，可以访问 http://localhost:8018
```

> **启动耗时说明**：首次启动需要 30-60 秒，用于遍历 49,425 场历史比赛初始化 336 支国家队的 Elo 等级分。后续所有预测都是毫秒级响应。

---

### 第六步：访问系统

| 入口 | 地址 |
|------|------|
| 🌐 **前端页面** | http://localhost:8018 |
| 📖 **API 文档** | http://localhost:8018/docs |
| ❤️ **健康检查** | http://localhost:8018/api/health |

---

### 开发/生产环境注意事项

| 事项 | 建议                                                             |
|------|----------------------------------------------------------------|
| **端口** | 默认 8018，可在 `app.py` 底部 `uvicorn.run()` 修改                      |
| **监听地址** | 默认 `0.0.0.0`（允许局域网访问），如仅本地使用改为 `127.0.0.1`                     |
| **生产部署** | 建议在 uvicorn 前挂一层反向代理 (nginx/Caddy)，并配置 HTTPS + CSP 头           |
| **Gunicorn** | 如需多进程：`gunicorn -k uvicorn.workers.UvicornWorker -w 4 app:app` |
| **模型文件** | 已加入 `.gitignore`，生产环境需在部署时重新训练或从安全存储拉取                         |
| **实时数据** | 前端每 30 分钟自动轮询 `/api/live`，无需额外配置                               |

---

### 重新训练模型

当 Kaggle 数据更新或需要调整特征后重新训练：

```bash
cd backend
python model/train.py
```

训练完成后自动更新 `backend/model/saved/` 下的模型文件，重启服务即可生效。

---

## 📊 模型性能

| 指标 | 数值 | 说明 |
|------|:----:|------|
| **胜平负准确率** | 57.0% | 三分类 Top-1 准确率（随机基线 33.3%） |
| **Log Loss** | 0.886 | 概率校准质量（越低越好） |
| **主队进球 MAE** | 1.03 | 平均绝对误差（实际进球 vs 预测进球） |
| **客队进球 MAE** | 0.84 | 同上 |
| **训练样本** | 19,673 场 | 2002—2023 时间段 |
| **测试样本** | 3,618 场 | 2023—2026 时间外验证 |

> 🎯 **行业基准参考**：FiveThirtyEight 世界杯预测准确率约 58-62%。本项目 57% 接近业界水平，且训练数据仅限 Kaggle 开放数据集，未使用球员身价、伤病等付费数据。

---

## 🔧 配置说明

### LLM 配置（前端管理面板）

打开前端页面 → 点击右上角齿轮图标 → LLM 配置面板：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| **厂商 (Provider)** | 预设 6 大厂商的 API 端点 | OpenAI / DeepSeek / Qwen / ... |
| **API Key** | 你的 LLM API Key（保存后写入 .env） | `sk-xxxx...` |
| **模型 (Model)** | 对应厂商的模型名称 | `gpt-4o-mini` / `deepseek-chat` |
| **自定义地址** | 自部署或第三方 OpenAI 兼容服务 | `http://your-server:8080/v1` |

> 配置保存在 `backend/data/llm_config.json`（不含 API Key）和 `.env`（仅含 API Key）两个文件中，做到了关注点分离。

### 配色方案变量

| CSS 变量 | 色值 | 用途 |
|----------|------|------|
| `--bg-primary` | `#0a0e27` | 主背景色 |
| `--accent-primary` | `#06b6d4` | 主强调色（青色） |
| `--accent-secondary` | `#8b5cf6` | 次强调色（紫色） |
| `--color-home-win` | `#10b981` | 主队胜绿色 |
| `--color-draw` | `#f59e0b` | 平局橙色 |
| `--color-away-win` | `#f43f5e` | 客队胜红色 |
| `--color-highlight` | `#fbbf24` | 高亮金色 |

---

## 🤔 已知限制与改进方向

| 限制 | 影响 | 计划 |
|------|------|------|
| **平局预测 F1=0.31** | 平局概率估计不准 | 行业共性问题；可引入贝叶斯层次模型 |
| **大比分估计偏低** | Poisson 对 4-0+ 概率拟合不足 | 考虑 Negative Binomial 或零膨胀模型 |
| **无球员维度** | 伤病、停赛、状态波动未纳入 | 需接入球员级数据源 |
| **无 CSP/限流** | 生产环境需补充安全头 | 挂 nginx 反向代理解决 |
| **时区硬编码** | 前端显示时间可能偏移 | 在数据显示层增加时区转换 |

---

## 📚 数据源

- **Kaggle**：[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — 49,425 场国际比赛原始数据
- **ESPN API**：`https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard` — 世界杯实时比分、赛程（无需 API Key）

---

## 📄 许可

本项目仅供学习和研究使用。预测结果不构成任何形式的投注建议。
