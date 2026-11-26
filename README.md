# Multi-Agent Hub(本地多 Agent 协调中心)

一个**本机版多 Agent 编排示例**:**Coordinator / Orchestrator-Workers(Router)模式**。

Hub(协调者)本身是一个由本地 Ollama 驱动的 Agent,它把 12 台「子 Agent」封装成工具。
你问一个问题,主 Agent 判断归属哪个领域、调用哪台或哪几台子 Agent,再把结果汇总成
一份完整回答。所有服务都在本机跑,数据源全部是免费公开 API,**无需任何云服务密钥**
(LLM 密钥都走环境变量注入)。

## 架构

```
┌─────────────────────── agents_hub.py · 主协调 Agent · :8910 ───────────────────────┐
│  路由① LLM 工具调用(Agently + Ollama qwen3:8b)                                      │
│  路由② 关键词规则路由(--fast,不依赖 LLM,秒回;LLM 失败时自动降级)                      │
│  网页端:子助手卡片 + 在线探测(/agents)+ 异步问答(提交 → 轮询 → 结果)                   │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┘
       │          │          │          │          │          │          │
   8892 地图    8900 赛事   8901 Comate 8902 股票   8903 直播   8904 天气   8905 NFL ··· 8914 星链 ··· 8915 时事新闻
   (每台子 Agent = 数据层 + 助手层两个脚本,详见下表)
```

## 子助手清单

| 端口 | 子助手 | 文件 | 数据源 |
|---|---|---|---|
| 8892 | 地图助手 | `map_assistant.py` | 高德地理编码 API、DataV 行政区划、Google/Baidu 地图链接 |
| 8900 | 赛事周报 | `sports_agent.py` + `sports_data.py` / `sports_rag.py` / `sports_report.py` | Jolpica F1、ESPN NBA/EPL/高尔夫 |
| 8901 | Comate 产品助手 | `comate_assistant.py` + `comate_rag.py` | 本地产品文档切片建 RAG(目录可用 `COMATE_DOCS_DIR` 覆盖) |
| 8902 | 股票助手 | `stock_agent.py` + `stock_data.py` | 新浪行情、雅虎财经 |
| 8903 | 直播衣物识别 | `live_detect.py` + `live_detect_data.py` | 本地规则识别(≥1000 元高价值品类) |
| 8904 | 天气助手 | `weather_agent.py` + `weather_data.py` | Open-Meteo 免费 API(未来 10 天) |
| 8905 | NFL 助手 | `nfl_agent.py` + `nfl_data.py` | ESPN NFL 接口 |
| 8906 | 汽车出口监测 | `auto_agent.py` + `auto_data.py` | UN Comtrade + 行业资讯(盖世汽车/汽车之家) |
| 8907 | 文档知识库 | `doc_kb_agent.py` + `doc_kb.py` | 本地文档扫描统计 + DashScope embedding RAG |
| 8908 | 记忆机器人 | `memory_bot.py` | Agently + Ollama,RecordStore 持久记忆 |
| 8914 | 星链卫星 | `starlink_agent.py` + `starlink_live.py` | CelesTrak 公开 TLE + sgp4 轨道传播(实时星下点分布图) |
| 8915 | 时事新闻 | `news_agent.py` + `news_data.py` | BBC 中文 / BBC World / 纽约时报中文 / Google News 中文 RSS + ECharts 可视化看板 |

> 数据索引/归档目录(`comate_docs_kb/`、`local_docs_kb/`、`bot_memory_data/`、
> `nfl_news_archive/`、`auto_export_data/`、`starlink_data/`、`news_assets/` 等)为运行时生成,已在 .gitignore 排除。

## 快速开始

要求:Python 3.11+;`ollama serve` 且已拉取一个 chat 模型(默认 `qwen3:8b`)。

```bash
pip install -r requirements.txt          # agently / flask / openai(子助手用)

# 终端 1:启动任意几台子助手(以天气 + 股票为例)
python weather_agent.py --serve --port 8904
python stock_agent.py   --serve --port 8902

# 终端 2:启动 Hub
python agents_hub.py --serve              # 打开 http://127.0.0.1:8910
```

网页端:顶部显示「子助手在线 x/12」,点击卡片可填该领域示例问题;输入问题点
「一键协调」,进度条轮询约 1~3 分钟后给出汇总回答。

## 命令行用法

```bash
python agents_hub.py "北京未来一周天气怎么样"   # 一次问答(主 Agent 路由)
python agents_hub.py --fast "北京天气和腾讯股价" # 规则路由,不依赖 LLM,秒回
python agents_hub.py --chat                    # 交互模式
python agents_hub.py --list                    # 列出子助手
python agents_hub.py --serve --port 8910       # HTTP 服务 + 网页
python agents_hub.py --acp                     # 作为 ACP Agent(stdio),供 ACP 客户端拉起
```

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/ask` | 同步问答 `{"question": "..."}` |
| POST | `/ask/async` | 异步提交,返回 `{"task_id": ...}` |
| GET | `/task/<id>` | 查询任务(状态/进度/结果) |
| GET | `/agents` | 子助手在线探测(后台并发探测各端口 `/health`) |
| GET | `/health` | 服务健康检查 |
| GET | `/` | 协调中心网页 |

## 配置(环境变量)

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | 主 Agent 的 OpenAI 兼容端点 |
| `LLM_MODEL` | `qwen3:8b` | 主 Agent 模型 |
| `LLM_API_KEY` | `ollama` | 对应端点密钥 |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | DashScope qwen-plus | 子助手读取同一组变量;都可切换到任意 OpenAI 兼容厂商(全本地:指到 Ollama) |
| `DASHSCOPE_API_KEY` | 无 | 子助手备选密钥(百炼 embedding/chat) |
| `AMAP_MAPS_API_KEY` | 无 | 地图助手的高德 key |
| `COMATE_DOCS_DIR` | `~/.comate/skills/.system/comate-docs` | Comate 助手文档目录 |
| `SPORTS_REPORT_DIR` | `~/SportsWeekly/app/sports_reports` | 赛事历史报告归档目录 |
| `BOT_MEMORY_DIR` | `./bot_memory_data` | 记忆机器人存储目录 |

## 如何扩展一台子助手

以「赛事周报」为例(把 `:8900` 换成你自己的服务即可):

1. **定义工具函数**(hub 端,封装本地 HTTP 服务;docstring 会被模型当作工具描述):

```python
@Agently.tool.action_func
def ask_sports(question: str) -> str:
    """查询综合体育赛事信息(赛程/比分/新闻等)。"""
    return _http_post(8900, "/ask", {"question": question})
```

2. **注册**:加入 `ALL_TOOLS`、`_TOOL_MAP`;想走规则路由再加一行 `_RULES`。
3. **上架**:在 `AGENT_ROWS` 加一行 `(端口, 名称, 工具名, 简介)`——网页卡片 / 在线探测 / `--list` 自动生效。

## 免责声明

- 数据来源于 Open-Meteo、ESPN、UN Comtrade、CelesTrak、新浪财经、雅虎财经、高德地图、盖世汽车、
  汽车之家等**公开接口**,仅供学习交流;请遵守各数据源的服务条款,勿用于商业用途
  或高频抓取。
- 本仓库不包含任何密钥、账号或内部文档内容;所有 LLM/地图密钥均通过环境变量注入,
  随附数据索引目录均被 .gitignore 排除。
- 本项目与所涉数据源、公司均无关联。

## License

[MIT](LICENSE)