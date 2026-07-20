"""LLM服务模块"""

import os
from hello_agents import HelloAgentsLLM
from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def get_llm() -> HelloAgentsLLM:
    """
    获取LLM实例(单例模式)

    Returns:
        HelloAgentsLLM实例
    """
    global _llm_instance

    if _llm_instance is None:
        get_settings()  # 确保已加载 backend/.env

        model = os.getenv("LLM_MODEL_ID")
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))

        # 显式传入项目配置，避免被 shell 中的 OPENAI_API_KEY(如 MiniMax)抢占
        provider = "auto"
        if base_url and "deepseek.com" in base_url.lower():
            provider = "deepseek"

        _llm_instance = HelloAgentsLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            timeout=timeout,
        )

        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")
        print(f"   Base URL: {_llm_instance.base_url}")
        print(f"   API Key 后缀: ...{(_llm_instance.api_key or '')[-4:]}")

    return _llm_instance
