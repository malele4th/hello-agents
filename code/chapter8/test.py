"""
第8章快速验证脚本（适配 hello-agents 1.0.0）

说明：
- 教材示例里的 MemoryTool / RAGTool 属于 0.2.x 教学版；
  在 1.0.0 中已移除（见 hello_agents.context.builder 注释）。
- 本脚本用 1.0 的会话历史 + CalculatorTool，验证 LLM 与 Agent 可正常运行。
- 同目录 .env 中的 Qdrant / Neo4j / Embedding 配置可留给自行实现记忆/RAG 时使用。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
from hello_agents.tools import CalculatorTool

# 加载本章 .env，并兼容 shell 里已有的 OPENAI_* / CORECODER_MODEL
load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
model = os.getenv("LLM_MODEL_ID") or os.getenv("CORECODER_MODEL")

if not all([api_key, base_url, model]):
    raise SystemExit(
        "缺少 LLM 配置。请在 code/chapter8/.env 设置 "
        "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID，"
        "或在环境中设置 OPENAI_API_KEY / OPENAI_BASE_URL / CORECODER_MODEL。"
    )

llm = HelloAgentsLLM(model=model, api_key=api_key, base_url=base_url)

tool_registry = ToolRegistry()
tool_registry.register_tool(CalculatorTool())

agent = SimpleAgent(
    name="智能助手",
    llm=llm,
    system_prompt=(
        "你是一个有用的AI助手。"
        "请记住用户在本轮对话中告诉你的个人信息，后续问题要能正确回忆。"
    ),
    tool_registry=tool_registry,
)

print(f"✅ hello-agents={__import__('hello_agents').__version__}, model={model}")
print("=== 第1轮：让助手记住信息 ===")
response1 = agent.run("你好！请记住我叫张三，我是一名Python开发者")
print(response1)

print("\n=== 第2轮：验证会话记忆 ===")
response2 = agent.run("我叫什么名字？我的职业是什么？")
print(response2)
