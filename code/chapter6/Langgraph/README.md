# LangGraph 智能搜索助手

本示例基于 **LangGraph 1.2.x**、LangChain 和 Tavily，实现一个具有状态管理、条件路由、失败重试与流式输出能力的智能搜索助手。

## 工作流程

```text
START → understand → search ─┬→ answer → END
                             └→ fallback → END
```

- `understand`：调用大模型理解用户需求并提炼搜索词。
- `search`：调用 Tavily 搜索最新资料。
- `answer`：搜索成功后，基于搜索结果生成答案。
- `fallback`：搜索失败后，使用大模型已有知识回答。

`SearchState` 保存节点之间共享的数据。`messages` 字段使用 LangGraph 的 `add_messages` reducer，因此节点返回的新消息会追加到历史记录，而不是覆盖原消息。

## 环境要求

- Python 3.10+
- LangGraph 1.2.9+
- 一个兼容 OpenAI Chat Completions API 的大模型服务
- Tavily API Key

安装依赖：

```bash
cd code/chapter6/Langgraph
pip install -r requirements.txt
```

项目会通过 `../load_env.py` 读取以下环境变量：

```bash
export OPENAI_API_KEY="你的大模型 API Key"
export OPENAI_BASE_URL="OpenAI 兼容接口地址"
export CORECODER_MODEL="模型名称"
export TAVILY_API_KEY="你的 Tavily API Key"
```

也可以使用对应的 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL_ID`。

## 运行方式

默认使用演示模式，自动询问“LangGraph 是什么？”：

```bash
python Dialogue_System.py
```

指定演示问题：

```bash
LANGGRAPH_DEMO_QUERY="Python 3.14 有哪些新特性？" python Dialogue_System.py
```

进入交互模式：

```bash
LANGGRAPH_DEMO=0 python Dialogue_System.py
```

输入 `quit`、`q`、`exit` 或 `退出` 可结束程序。

## LangGraph 1.2.x 用法

### 状态与 reducer

```python
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: str
    search_results: str
    step: str
```

每个节点接收当前状态，只返回自己需要更新的字段，LangGraph 负责合并状态。

### 条件路由

搜索结束后，路由函数根据 `step` 决定下一个节点：

```python
def route_after_search(state):
    if state["step"] == "search_failed":
        return "fallback"
    return "answer"
```

### 节点重试

搜索节点配置了重试策略：

```python
RetryPolicy(max_attempts=3, initial_interval=1.0)
```

网络连接或超时异常会交给 LangGraph 自动重试；其他 API 或业务错误会进入 `fallback` 节点。

### v2 流式事件

```python
async for chunk in app.astream(
    initial_state,
    config=config,
    stream_mode="updates",
    version="v2",
):
    ...
```

`stream_mode="updates"` 表示每个节点完成后返回该节点产生的状态增量，而不是等待整张图运行结束。

### Checkpoint

图使用 `InMemorySaver` 保存运行状态：

```python
workflow.compile(checkpointer=InMemorySaver())
```

运行时必须通过 `thread_id` 标识会话：

```python
config = {"configurable": {"thread_id": "search-demo"}}
```

`InMemorySaver` 只适合教学和本地演示；生产环境应使用数据库支持的持久化 checkpointer。

## Python 异步语法

### `async def`

`async def` 用于定义异步函数，也称为协程函数：

```python
async def invoke_llm(system: str, user: str):
    response = await llm.ainvoke(...)
    return response
```

调用普通函数会立即执行并返回结果；调用异步函数会得到一个协程对象，需要通过 `await` 或事件循环执行。

异步函数适合以下 I/O 操作：

- 调用大模型 API
- 网络搜索
- 数据库请求
- 读取远程文件

等待这些操作时，事件循环可以调度其他任务，避免整个程序被同步等待阻塞。

### `await`

`await` 只能直接写在 `async def` 中，用于等待异步操作完成：

```python
response = await llm.ainvoke(messages)
```

执行到 `await` 时，当前协程暂时让出控制权；等待结束后，再从此处继续执行。

### `asyncio.to_thread`

Tavily SDK 的 `search()` 是同步函数。为了避免它阻塞异步事件循环，程序将它放在线程池中执行：

```python
response = await asyncio.to_thread(
    tavily_client.search,
    query=search_query,
)
```

### `asyncio.run(main())`

Python 脚本入口本身是同步环境，而 `main()` 是异步函数：

```python
async def main():
    await run_search_workflow(...)


if __name__ == "__main__":
    asyncio.run(main())
```

`asyncio.run(main())` 会：

1. 创建事件循环；
2. 执行 `main()` 协程直到结束；
3. 关闭事件循环并清理资源。

它是 Python 3.7+ 在普通脚本中启动异步主函数的推荐方式。

## 异步调用链

```text
asyncio.run(main())
  └─ await run_search_workflow()
      └─ app.astream()
          ├─ understand_query_node()
          │   └─ await llm.ainvoke()
          ├─ tavily_search_node()
          │   └─ await asyncio.to_thread(...)
          └─ generate_answer_node()
              └─ await llm.ainvoke()
```

一句话记忆：

- `async def`：定义可以暂停等待的异步函数。
- `await`：等待异步操作完成，同时允许事件循环处理其他任务。
- `asyncio.run(main())`：从普通 Python 脚本启动整个异步程序。
