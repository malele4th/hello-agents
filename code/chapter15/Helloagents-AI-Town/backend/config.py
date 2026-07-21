"""配置文件"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 项目 .env 优先于 shell 全局 OPENAI_*，避免串用其他章节的 Key
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


class Settings:
    """应用配置"""

    # API配置
    API_TITLE = "赛博小镇 API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # NPC配置
    NPC_UPDATE_INTERVAL = int(os.getenv("NPC_UPDATE_INTERVAL", "30"))

    # LLM配置 (从环境变量读取)
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "deepseek-v4-flash")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")

    # CORS配置
    CORS_ORIGINS = ["*"]  # 生产环境应限制具体域名

    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.LLM_API_KEY:
            print("⚠️  警告: 未设置LLM_API_KEY环境变量")
            print("   请在.env文件中配置LLM_API_KEY")
            print("   示例: LLM_API_KEY=\"your-api-key\"")
            return False

        print(f"✅ LLM配置:")
        print(f"   提供商: {cls.LLM_PROVIDER}")
        print(f"   模型: {cls.LLM_MODEL_ID}")
        print(f"   服务地址: {cls.LLM_BASE_URL}")
        print(f"   API Key 后缀: ...{(cls.LLM_API_KEY or '')[-4:]}")
        return True


settings = Settings()
