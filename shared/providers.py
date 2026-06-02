"""Built-in presets for common OpenAI-compatible LLM providers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    base_url: str
    default_model: str
    models: tuple[str, ...]
    api_key_env: str = "OPENAI_API_KEY"
    api_key_optional: bool = False


PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
        api_key_env="OPENAI_API_KEY",
    ),
    "deepseek": Provider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "moonshot": Provider(
        id="moonshot",
        name="Moonshot (Kimi)",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        models=("moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"),
        api_key_env="MOONSHOT_API_KEY",
    ),
    "zhipu": Provider(
        id="zhipu",
        name="Zhipu (GLM)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        models=("glm-4-flash", "glm-4-air", "glm-4"),
        api_key_env="ZHIPU_API_KEY",
    ),
    "dashscope": Provider(
        id="dashscope",
        name="Alibaba DashScope (Qwen)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-turbo",
        models=("qwen-turbo", "qwen-plus", "qwen-max"),
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "siliconflow": Provider(
        id="siliconflow",
        name="SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
        default_model="deepseek-ai/DeepSeek-V3",
        models=(
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        api_key_env="SILICONFLOW_API_KEY",
    ),
    "groq": Provider(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        api_key_env="GROQ_API_KEY",
    ),
    "together": Provider(
        id="together",
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        models=("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",),
        api_key_env="TOGETHER_API_KEY",
    ),
    "openrouter": Provider(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        models=("openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"),
        api_key_env="OPENROUTER_API_KEY",
    ),
    "ollama": Provider(
        id="ollama",
        name="Ollama (local)",
        base_url="http://localhost:11434/v1",
        default_model="llama3.2",
        models=("llama3.2", "qwen2.5"),
        api_key_env="OPENAI_API_KEY",
        api_key_optional=True,
    ),
}

DEFAULT_PROVIDER_ID = "openai"


def get_provider(provider_id: str) -> Provider:
    key = provider_id.lower().strip()
    if key not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider: {provider_id}. Available: {known}")
    return PROVIDERS[key]


def list_provider_ids() -> list[str]:
    return sorted(PROVIDERS)
