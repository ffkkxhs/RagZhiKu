# atguigu/query_process/nodes/node_answer_output.py
import os
import re

from langchain.chat_models import init_chat_model

from atguigu.config.config import LLMConfig
from atguigu.config.prompt import ANSWER_PROMPT
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
from atguigu.tool.mongo_tool import save_chat_message
from atguigu.tool.task_utils import put_data


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 所有字段都做空值兜底：这个节点既可能走"拒答直通"，也可能走"正常生成"
        answer = state.get("answer") or ""
        task_id = state.get("task_id")
        history_list = state.get("history") or []
        # 优先使用父块扩展后的文档；没有扩展节点时退回 reranked_docs
        reranked_docs = state.get("expanded_docs") or state.get("reranked_docs") or []
        item_names = state.get("item_names") or []
        rewritten_query = state.get("rewritten_query") or ""
        session_id = state.get("session_id")

        # 评估的 retrieval 模式只需要 reranked_docs，不需要花 token 生成答案。
        # run_eval.py 会设置 EVAL_SKIP_ANSWER=1；生产/手动调用不受影响。
        if os.getenv("EVAL_SKIP_ANSWER") == "1":
            return {"answer": answer}

        # 拒答/待选词已经在商品名确认节点生成过，这里只负责推给前端，不再调模型
        message_id = None
        if answer:
            put_data(task_id, event="final", data={"answer": answer})
            # 把 answer 一起返回：流式调用方继续看 SSE，
            # 评估脚本/非流式调用方可以直接从 state 里拿最终答案。
            return {"answer": answer, "message_id": message_id}

        # 1. 把重排后的资料拼成带编号的上下文。
        #    编号是为了让模型能引用 [1]、[2]，也方便后续做引用校验。
        content_parts = []
        for idx, doc in enumerate(reranked_docs, start=1):
            content_parts.append(
                # 优先用 title_path，引用能精确到“文档 > 章节 > 小节”
                f"[{idx}] 标题：{doc.get('title_path') or doc.get('title') or ''}\n"
                f"来源：{doc.get('source') or ''} {doc.get('url') or ''}\n"
                f"内容：{doc.get('content') or ''}"
            )
        # 只截断资料，不截断整个 prompt，避免把最后的问题和"请回答"截掉
        content_from_chunks = "\n\n".join(content_parts)[:10000]

        # 2. Mongo 返回的历史是倒序，拼 prompt 前转回时间正序
        history_parts = []
        for item in reversed(history_list):
            history_parts.append(f"{item.get('role')}:{item.get('text')}")
        history = "\n".join(history_parts)

        # 3. 参数名必须和 ANSWER_PROMPT 里的 {question}/{context}/{item_names}/{history} 对齐
        prompt = ANSWER_PROMPT.format(
            question=rewritten_query,
            context=content_from_chunks,
            item_names=",".join(item_names),
            history=history,
        )

        # 4. 流式生成，逐 token 推 delta 事件
        llm = init_chat_model(
            model=LLMConfig.llm_default_model,
            model_provider="openai",
            api_key=LLMConfig.openai_api_key,
            base_url=LLMConfig.openai_api_base,
            temperature=LLMConfig.llm_default_temperature
        )

        for chunk in llm.stream(prompt):
            delta = chunk.content or ""
            if not delta:
                continue
            answer += delta
            put_data(task_id, event="delta", data={"delta": delta})

        # 5. 从参考内容里收集图片 URL，去重且保持出现顺序
        image_list = []
        seen_images = set()
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        for doc in reranked_docs:
            for img_url in md_img_pattern.findall(doc.get("content") or ""):
                img_url = img_url.strip()
                if img_url and img_url not in seen_images:
                    seen_images.add(img_url)
                    image_list.append(img_url)

        # final 同时带上完整答案和图片列表，前端即使没拼完 delta 也能拿到最终文本
        put_data(task_id, event="final", data={"answer": answer, "image_urls": image_list})

        # 6. 只有真正生成了答案才写 Mongo，避免空消息
        if answer and session_id:
            message_id = save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )

        # 同上：评估和调试时需要拿到完整答案，而不仅是 message_id
        return {
            "answer": answer,
            "message_id": message_id,
        }

