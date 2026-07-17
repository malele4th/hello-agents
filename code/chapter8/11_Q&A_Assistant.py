#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档问答助手 - 基于 HelloAgents 1.0 的智能文档问答系统

适配说明：
- 1.0 已移除 MemoryTool / RAGTool
- 本文件改用 qa_backend.py（本地记忆 + Embedding RAG + HelloAgentsLLM）
- Gradio UI 交互流程与教材一致
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from qa_backend import SimpleMemoryStore, SimpleRAGEngine


class PDFLearningAssistant:
    """智能文档问答助手（hello-agents 1.0 兼容实现）"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.memory_tool = SimpleMemoryStore(user_id=user_id)
        self.rag_tool = SimpleRAGEngine(namespace=f"pdf_{user_id}")

        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }
        self.current_document = None

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}

        start_time = time.time()
        try:
            result = self.rag_tool.add_document(
                file_path=pdf_path,
                chunk_size=1000,
                chunk_overlap=200,
            )
            process_time = time.time() - start_time

            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            self.memory_tool.add(
                content=f"加载了文档《{self.current_document}》",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id,
            )

            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒) {result}",
                "document": self.current_document,
            }
        except Exception as e:
            return {"success": False, "message": f"加载失败: {str(e)}"}

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        if not self.current_document:
            return "⚠️ 请先加载文档！使用 load_document() 方法加载PDF文档。"

        self.memory_tool.add(
            content=f"提问: {question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id,
        )

        answer = self.rag_tool.ask(
            question=question,
            limit=5,
            enable_advanced_search=use_advanced_search,
        )

        self.memory_tool.add(
            content=f"关于'{question}'的学习",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id,
        )
        self.stats["questions_asked"] += 1
        return answer

    def add_note(self, content: str, concept: Optional[str] = None):
        self.memory_tool.add(
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id,
        )
        self.stats["concepts_learned"] += 1

    def recall(self, query: str, limit: int = 5) -> str:
        return self.memory_tool.search(query=query, limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载",
            "RAG状态": self.rag_tool.stats(),
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        memory_summary = self.memory_tool.summary(limit=10)
        rag_stats = self.rag_tool.stats()
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration,
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats,
        }

        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            try:
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                report["report_file"] = report_file
            except Exception as e:
                report["save_error"] = str(e)
        return report


def create_gradio_ui():
    """创建Gradio Web UI"""
    assistant_state = {"assistant": None}

    def init_assistant(user_id: str) -> str:
        if not user_id:
            user_id = "web_user"
        assistant_state["assistant"] = PDFLearningAssistant(user_id=user_id)
        return f"✅ 助手已初始化 (用户: {user_id}, hello-agents 1.0)"

    def load_pdf(pdf_file) -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        if pdf_file is None:
            return "❌ 请上传PDF文件"

        pdf_path = pdf_file if isinstance(pdf_file, str) else getattr(pdf_file, "name", None)
        if not pdf_path:
            return "❌ 无法识别上传的文件路径"

        result = assistant_state["assistant"].load_document(pdf_path)
        if result["success"]:
            return f"✅ {result['message']}\n📄 文档: {result['document']}"
        return f"❌ {result['message']}"

    def chat(message: str, history: List) -> Tuple[str, List]:
        history = list(history or [])

        def _append(user_msg: str, bot_msg: str):
            # Gradio 5+/6 默认 messages 格式；兼容旧 tuples 格式
            if history and isinstance(history[0], dict):
                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": bot_msg})
            elif not history:
                # 新版 Gradio 优先 messages
                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": bot_msg})
            else:
                history.append([user_msg, bot_msg])

        if assistant_state["assistant"] is None:
            _append(message, "❌ 请先初始化助手并加载文档")
            return "", history

        if not message.strip():
            return "", history

        if any(keyword in message for keyword in ["之前", "学过", "回顾", "历史", "记得"]):
            response = assistant_state["assistant"].recall(message)
            response = f"🧠 **学习回顾**\n\n{response}"
        else:
            response = assistant_state["assistant"].ask(message)
            response = f"💡 **回答**\n\n{response}"

        _append(message, response)
        return "", history

    def add_note_ui(note_content: str, concept: str) -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        if not note_content.strip():
            return "❌ 笔记内容不能为空"
        assistant_state["assistant"].add_note(note_content, concept or None)
        return f"✅ 笔记已保存: {note_content[:50]}..."

    def get_stats_ui() -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        stats = assistant_state["assistant"].get_stats()
        result = "📊 **学习统计**\n\n"
        for key, value in stats.items():
            result += f"- **{key}**: {value}\n"
        return result

    def generate_report_ui() -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        report = assistant_state["assistant"].generate_report(save_to_file=True)
        result = "✅ 学习报告已生成\n\n"
        result += "**会话信息**\n"
        result += f"- 会话时长: {report['session_info']['duration_seconds']:.0f}秒\n"
        result += f"- 加载文档: {report['learning_metrics']['documents_loaded']}\n"
        result += f"- 提问次数: {report['learning_metrics']['questions_asked']}\n"
        result += f"- 学习笔记: {report['learning_metrics']['concepts_learned']}\n"
        if "report_file" in report:
            result += f"\n💾 报告已保存至: {report['report_file']}"
        return result

    with gr.Blocks(title="智能文档问答助手") as demo:
        gr.Markdown(
            """
        # 📚 智能文档问答助手

        基于 HelloAgents **1.0** 的智能文档问答系统：
        - 📄 加载PDF并构建知识库（Embedding RAG）
        - 💬 智能问答
        - 📝 学习笔记 / 🧠 历程回顾 / 📊 学习报告
        """
        )

        with gr.Tab("🏠 开始使用"):
            with gr.Row():
                user_id_input = gr.Textbox(
                    label="用户ID",
                    placeholder="输入你的用户ID（可选，默认为web_user）",
                    value="web_user",
                )
                init_btn = gr.Button("初始化助手", variant="primary")
            init_output = gr.Textbox(label="初始化状态", interactive=False)
            init_btn.click(init_assistant, inputs=[user_id_input], outputs=[init_output])

            gr.Markdown("### 📄 加载PDF文档")
            pdf_upload = gr.File(label="上传PDF文件", file_types=[".pdf"], type="filepath")
            load_btn = gr.Button("加载文档", variant="primary")
            load_output = gr.Textbox(label="加载状态", interactive=False)
            load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

        with gr.Tab("💬 智能问答"):
            gr.Markdown("### 向文档提问或回顾学习历程")
            chatbot = gr.Chatbot(label="对话历史", height=400)
            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：什么是Transformer？ 或 我之前学过什么？",
                    scale=4,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "什么是大语言模型？",
                    "Transformer架构有哪些核心组件？",
                    "如何训练大语言模型？",
                    "我之前学过什么内容？",
                    "回顾一下关于注意力机制的学习",
                ],
                inputs=msg_input,
            )
            msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

        with gr.Tab("📝 学习笔记"):
            gr.Markdown("### 记录学习心得和重要概念")
            note_content = gr.Textbox(label="笔记内容", placeholder="输入你的学习笔记...", lines=3)
            concept_input = gr.Textbox(label="相关概念（可选）", placeholder="例如：transformer, attention")
            note_btn = gr.Button("保存笔记", variant="primary")
            note_output = gr.Textbox(label="保存状态", interactive=False)
            note_btn.click(add_note_ui, inputs=[note_content, concept_input], outputs=[note_output])

        with gr.Tab("📊 学习统计"):
            gr.Markdown("### 查看学习进度和统计信息")
            stats_btn = gr.Button("刷新统计", variant="primary")
            stats_output = gr.Markdown()
            stats_btn.click(get_stats_ui, outputs=[stats_output])

            gr.Markdown("### 生成学习报告")
            report_btn = gr.Button("生成报告", variant="primary")
            report_output = gr.Textbox(label="报告状态", interactive=False)
            report_btn.click(generate_report_ui, outputs=[report_output])

    return demo


def main():
    print("\n" + "=" * 60)
    print("智能文档问答助手 (hello-agents 1.0)")
    print("=" * 60)
    print("正在启动Web界面...\n")
    print("提示: QDRANT_URL 若仍是占位符，将自动使用本地向量库。")
    print("访问: http://127.0.0.1:7860\n")

    # 规避部分环境下 Gradio 误判 localhost 不可达
    try:
        import gradio.networking as networking

        networking.url_ok = lambda _url: True  # type: ignore[method-assign]
    except Exception:
        pass

    # 规避 gradio_client 在解析某些 JSON Schema 时的 TypeError
    try:
        import gradio_client.utils as client_utils

        _orig_get_type = client_utils.get_type

        def _safe_get_type(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)

        client_utils.get_type = _safe_get_type  # type: ignore[assignment]
    except Exception:
        pass

    demo = create_gradio_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
