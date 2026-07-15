import os
from pathlib import Path

from dotenv import load_dotenv


def load_chapter6_env() -> dict[str, str]:
    """Load LLM env vars, preferring LLM_* and falling back to shell config."""
    chapter6_dir = Path(__file__).resolve().parent
    load_dotenv(chapter6_dir / ".env")
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = (
        os.getenv("LLM_MODEL_ID")
        or os.getenv("CORECODER_MODEL")
        or os.getenv("LLM_MODEL")
    )
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    if api_key:
        os.environ.setdefault("LLM_API_KEY", api_key)
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("LLM_BASE_URL", base_url)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
    if model:
        os.environ.setdefault("LLM_MODEL_ID", model)
        os.environ.setdefault("CORECODER_MODEL", model)
        os.environ.setdefault("LLM_MODEL", model)

    return {
        "api_key": api_key or "",
        "base_url": base_url or "",
        "model": model or "",
        "tavily_api_key": tavily_api_key or "",
        "serpapi_api_key": serpapi_api_key or "",
    }


def require_llm_env() -> dict[str, str]:
    env = load_chapter6_env()
    missing = [name for name in ("api_key", "base_url", "model") if not env[name]]
    if missing:
        raise ValueError(
            "缺少 LLM 配置: "
            + ", ".join(missing)
            + "。请设置 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL_ID，"
            "或 OPENAI_API_KEY/OPENAI_BASE_URL/CORECODER_MODEL。"
        )
    return env
