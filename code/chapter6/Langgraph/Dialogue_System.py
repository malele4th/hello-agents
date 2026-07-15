"""
智能搜索助手 - 基于 LangGraph 1.2.x + Tavily API 的真实搜索系统

工作流：
1. 理解用户需求并生成搜索词
2. 使用 Tavily API 搜索
3. 基于搜索结果生成回答（搜索失败时走兜底节点）

图结构：
    START → understand → search ─┬→ answer → END
                                 └→ fallback → END
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from tavily import TavilyClient

# 复用 chapter6 统一的环境变量加载（OPENAI_* / CORECODER_MODEL / TAVILY_API_KEY）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_env import require_llm_env

env = require_llm_env()

# LLM 系统角色：各节点共用，具体任务通过 HumanMessage 传入
SEARCH_ASSISTANT_SYSTEM = (
    "你是一个智能搜索助手，擅长理解用户问题、提炼搜索关键词，"
    "并基于搜索结果给出准确、结构清晰的回答。"
)

# 搜索节点重试策略：网络抖动时自动重试，业务错误在节点内处理不走重试
SEARCH_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=1.0)


class SearchState(TypedDict):
    """LangGraph 图状态：节点返回的字典会与当前状态合并。"""

    # add_messages 是内置 reducer，新消息会追加而非覆盖整个列表
    messages: Annotated[list, add_messages]
    user_query: str        # LLM 对用户需求的理解摘要
    search_query: str      # 提炼后的 Tavily 搜索关键词
    search_results: str      # Tavily 返回的格式化文本
    final_answer: str      # 最终回答（便于后续扩展持久化或评测）
    step: str              # 流程标记，供条件路由使用（understood / searched / search_failed / completed）


# --- 模型与外部服务初始化 ---

llm = ChatOpenAI(
    model=env["model"],
    api_key=env["api_key"],
    base_url=env["base_url"],
    temperature=0.7,
)
tavily_client = TavilyClient(api_key=env["tavily_api_key"])


# --- LLM 与工具函数 ---

async def invoke_llm(system: str, user: str) -> AIMessage:
    """异步调用 LLM。

    同时发送 SystemMessage + HumanMessage，兼容 MiniMax 等
    不接受「仅有 system、没有 user」消息的 OpenAI 兼容接口。
    """
    response = await llm.ainvoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )
    return response


def extract_latest_user_message(messages: list) -> str:
    """从消息列表中取最近一条用户输入（支持多轮对话场景）。"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def extract_search_query(response_text: str, fallback: str) -> str:
    """从 LLM 结构化输出中解析搜索词，解析失败则回退到原始用户问题。"""
    if "搜索词：" in response_text:
        return response_text.split("搜索词：", 1)[1].strip()
    if "搜索关键词：" in response_text:
        return response_text.split("搜索关键词：", 1)[1].strip()
    return fallback


def format_tavily_results(response: dict) -> str:
    """将 Tavily JSON 响应整理为可供 LLM 阅读的纯文本。"""
    search_results = ""

    if response.get("answer"):
        search_results = f"综合答案：\n{response['answer']}\n\n"

    if response.get("results"):
        search_results += "相关信息：\n"
        for index, result in enumerate(response["results"][:3], 1):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            search_results += f"{index}. {title}\n{content}\n来源：{url}\n\n"

    return search_results or "抱歉，没有找到相关信息。"


# --- LangGraph 节点 ---

async def understand_query_node(state: SearchState) -> dict:
    """节点 1：理解用户查询并生成搜索关键词。"""
    user_message = extract_latest_user_message(state["messages"])
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = await invoke_llm(SEARCH_ASSISTANT_SYSTEM, understand_prompt)
    response_text = response.content

    return {
        "user_query": response_text,
        "search_query": extract_search_query(response_text, user_message),
        "step": "understood",
        "messages": [AIMessage(content=f"我理解您的需求：{response_text}")],
    }


async def tavily_search_node(state: SearchState) -> dict:
    """节点 2：调用 Tavily 执行真实网页搜索。"""
    search_query = state["search_query"]
    print(f"🔍 正在搜索: {search_query}")

    try:
        # Tavily SDK 是同步的，放到线程池避免阻塞 asyncio 事件循环
        response = await asyncio.to_thread(
            tavily_client.search,
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5,
        )
    except (ConnectionError, TimeoutError):
        # 网络类异常向上抛出，由节点级 RetryPolicy 自动重试
        raise
    except Exception as error:
        # 业务/API 类错误不重试，标记 step 后走 fallback 分支
        error_msg = f"搜索时发生错误: {error}"
        print(f"❌ {error_msg}")
        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [
                AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")
            ],
        }

    return {
        "search_results": format_tavily_results(response),
        "step": "searched",
        "messages": [
            AIMessage(content="✅ 搜索完成！找到了相关信息，正在为您整理答案...")
        ],
    }


def route_after_search(state: SearchState) -> Literal["answer", "fallback"]:
    """条件边路由：根据搜索是否成功决定下一节点。"""
    if state.get("step") == "search_failed":
        return "fallback"
    return "answer"


async def generate_answer_node(state: SearchState) -> dict:
    """节点 3a：基于 Tavily 搜索结果生成最终答案。"""
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    response = await invoke_llm(SEARCH_ASSISTANT_SYSTEM, answer_prompt)
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


async def fallback_answer_node(state: SearchState) -> dict:
    """节点 3b：搜索失败时，仅用 LLM 内置知识兜底回答。"""
    fallback_prompt = f"""搜索 API 暂时不可用，请基于你的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""

    response = await invoke_llm(SEARCH_ASSISTANT_SYSTEM, fallback_prompt)
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


# --- 图构建与运行 ---

def create_search_assistant():
    """编译 LangGraph 工作流，返回可 invoke/astream 的应用对象。"""
    workflow = StateGraph(SearchState)

    workflow.add_node("understand", understand_query_node)
    workflow.add_node(
        "search",
        tavily_search_node,
        retry_policy=SEARCH_RETRY_POLICY,  # LangGraph 1.2+ 节点级重试
    )
    workflow.add_node("answer", generate_answer_node)
    workflow.add_node("fallback", fallback_answer_node)

    # 线性边 + 条件边
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("answer", END)
    workflow.add_edge("fallback", END)

    # InMemorySaver 支持 thread_id 维度的会话状态隔离（本 demo 用于演示 checkpoint 能力）
    return workflow.compile(checkpointer=InMemorySaver())


def print_node_update(node_name: str, node_output: dict) -> None:
    """解析单个节点的状态更新并打印用户可读进度。"""
    messages = node_output.get("messages") or []
    if not messages:
        return

    latest_message = messages[-1]
    if not isinstance(latest_message, AIMessage):
        return

    if node_name == "understand":
        print(f"🧠 理解阶段: {latest_message.content}")
    elif node_name == "search":
        print(f"🔍 搜索阶段: {latest_message.content}")
    elif node_name in {"answer", "fallback"}:
        print(f"\n💡 最终回答:\n{latest_message.content}")


async def run_search_workflow(app, initial_state: dict, config: dict) -> None:
    """以 LangGraph 1.2 v2 事件流方式执行图，逐节点输出更新。"""
    async for chunk in app.astream(
        initial_state,
        config=config,
        stream_mode="updates",  # 每个节点完成后推送一次状态增量
        version="v2",             # 新版流式事件格式
    ):
        if chunk.get("type") != "updates":
            continue
        for node_name, node_output in chunk["data"].items():
            print_node_update(node_name, node_output)


def build_initial_state(user_input: str) -> dict:
    """构造图的初始状态，其余字段由后续节点逐步填充。"""
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_query": "",
        "search_query": "",
        "search_results": "",
        "final_answer": "",
        "step": "start",
    }


async def main() -> None:
    if not env["tavily_api_key"]:
        print("❌ 错误：请配置 TAVILY_API_KEY")
        return

    app = create_search_assistant()
    demo_query = os.getenv("LANGGRAPH_DEMO_QUERY", "LangGraph 是什么？")

    # LANGGRAPH_DEMO=1 时自动跑一条示例问题，便于 CI/教学演示
    if os.getenv("LANGGRAPH_DEMO", "1") == "1":
        print("🔍 智能搜索助手演示模式启动...")
        print(f"🤔 演示问题: {demo_query}\n")
        await run_search_workflow(
            app,
            build_initial_state(demo_query),
            config={"configurable": {"thread_id": "search-demo"}},
        )
        print("\n✅ LangGraph 演示完成")
        return

    # 交互模式：每次提问使用独立 thread_id，互不影响 checkpoint
    print("🔍 智能搜索助手启动！")
    print("我会使用 Tavily API 为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    session_count = 0
    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        if user_input.lower() in {"quit", "q", "退出", "exit"}:
            print("感谢使用！再见！👋")
            break
        if not user_input:
            continue

        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}

        try:
            print("\n" + "=" * 60)
            await run_search_workflow(app, build_initial_state(user_input), config)
            print("\n" + "=" * 60 + "\n")
        except Exception as error:
            print(f"❌ 发生错误: {error}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    asyncio.run(main())
