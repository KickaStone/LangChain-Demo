# A Simple LangChain Demo



使用uv管理python环境和包

简单LangChain Demo

- 实现简单的助理问答

- 通过环境变量配置openai api key
- 支持.env文件配置
- 模型选择消耗较小的做实验

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key（二选一）
cp .env.example .env   # 编辑 .env 填入 OPENAI_API_KEY
# 或 export OPENAI_API_KEY=sk-...

# 3. 启动对话
uv run python main.py
```

默认使用 `gpt-4o-mini` 模型，可通过环境变量 `OPENAI_MODEL` 覆盖。


