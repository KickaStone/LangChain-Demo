# LLM Demos

使用 uv 管理 Python 环境与依赖。两个 demo 共用项目根目录的 `.env` 配置。

## 项目结构

```
.
├── .env                 # 共享配置（勿提交）
├── .env.example
├── shared/              # 共享：提供商预设、环境变量加载
├── langchain_demo/      # LangChain 对话 demo
├── langgraph_demo/      # LangGraph 对话 demo（带 checkpoint 记忆）
└── langgraph.json       # LangGraph Studio 配置
```

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key
cp .env.example .env   # 编辑 .env

# 3. 运行 demo（任选其一）
uv run langchain-demo
uv run langgraph-demo

# 或直接运行模块
uv run python -m langchain_demo.main
uv run python -m langgraph_demo.main
```

## 共享环境变量

| 变量 | 说明 |
| --- | --- |
| `LLM_PROVIDER` | 内置 API 提供商（默认 `openai`） |
| `OPENAI_API_KEY` | API Key（也支持各厂商专用变量） |
| `OPENAI_MODEL` | 默认模型 |
| `OPENAI_MODELS` | 可切换模型列表（逗号分隔） |
| `OPENAI_BASE_URL` | 自定义 API 地址 |
| `OPENAI_TEMPERATURE` | 采样温度 |

内置提供商：`openai`、`deepseek`、`moonshot`、`zhipu`、`dashscope`、`siliconflow`、`groq`、`together`、`openrouter`、`ollama`。

对话中命令：`/providers`、`/provider <id>`、`/models`、`/model <name>`、`/help`。

## Demo 说明

### LangChain Demo

基于 `RunnableWithMessageHistory` 的多轮对话，适合学习 LangChain 链式调用。

### LangGraph Demo

基于 `StateGraph` + `InMemorySaver` 的多轮对话，通过 checkpoint 持久化会话状态，适合学习 LangGraph 图结构与状态管理。

### LangGraph Studio

在 `.env` 中配置 `LANGSMITH_API_KEY`（在 [smith.langchain.com](https://smith.langchain.com) 创建），然后：

```bash
uv sync --group dev
uv run langgraph dev
```

浏览器打开终端输出的 Studio 链接，例如：

`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

Safari 若无法连接 localhost，使用 `uv run langgraph dev --tunnel`。

静态查看图结构（无需 Studio）：

```bash
uv run python -m langgraph_demo.visualize
```

## 配置示例

DeepSeek：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
```

本地 Ollama：

```bash
LLM_PROVIDER=ollama
OPENAI_MODEL=llama3.2
```
