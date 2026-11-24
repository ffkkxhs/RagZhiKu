# atguigu/query_process/nodes/node_parent_expand.py
"""
父子切块的查询侧节点：命中子块后回捞父块。

# ============================================================================
# 【改造说明：真正的父块回捞】
#
# 之前只在子 chunk 上存了 parent_id，但没有地方使用它。
# 这个节点在 rerank 之后执行：
# 1. 收集 Top-K 子块的 parent_id；
# 2. 从 Mongo 的 doc_parents 集合批量读取父块；
# 3. 同一个父块下的多个子块只保留一次（去重）；
# 4. 用父块正文替换子块正文，交给答案节点。
#
# 这样：
# - 检索仍然用小块，保证精度；
# - 生成用整节，保证上下文完整；
# - 对比题/长章节不会再因为 chunk 太碎而丢上下文。
# ============================================================================
"""

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
from atguigu.tool.mongo_tool import get_parents

# 最多回捞多少个父块，避免 prompt 过长
MAX_PARENT_DOCS = 8
# 单个父块正文的最大字符数（父块在导入时已经截断过一次）
MAX_PARENT_CONTENT_CHARS = 8000


class NodeParentExpand(NodeBase):
    """
    节点功能：把子块命中扩展成父块上下文。
    """

    name: str = "node_parent_expand"

    def process(self, state: QueryGraphState):
        """
        参数：
        - state: 查询图状态，读取 reranked_docs

        返回：
        - {"expanded_docs": 父块扩展后的文档列表}
        """
        docs = state.get("reranked_docs") or []

        if not docs:
            return {"expanded_docs": []}

        # 收集需要回捞的 parent_id（去重、保持顺序）
        parent_ids = []
        for doc in docs:
            parent_id = doc.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        # 没有 parent_id（例如 web 文档）时，原样返回
        if not parent_ids:
            return {"expanded_docs": docs}

        try:
            parents = get_parents(parent_ids)
        except Exception as error:
            # 父块读取失败不影响回答，降级为子块
            logger.warning(f"父块回捞失败，降级使用子块: {error}")
            return {"expanded_docs": docs}

        parent_map = {
            parent.get("parent_id"): parent
            for parent in parents
            if parent.get("parent_id")
        }

        expanded = []
        seen_parents = set()

        for doc in docs:
            parent_id = doc.get("parent_id")
            parent = parent_map.get(parent_id)

            if not parent:
                # 没找到父块（或本来就是 web 文档），保留子块
                expanded.append(doc)
                continue

            # 同一个父块下的多个子块只保留一次
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            content = (parent.get("content") or "").strip()
            if not content:
                content = doc.get("content") or ""

            expanded.append({
                **doc,
                "parent_id": parent_id,
                "title": parent.get("title_path") or doc.get("title") or "",
                "title_path": parent.get("title_path") or doc.get("title_path") or "",
                "content": content[:MAX_PARENT_CONTENT_CHARS],
                # 父块带同一份标签；老数据没有 labels 时退回子块的
                "labels": parent.get("labels") or doc.get("labels") or {},
                "item_name": parent.get("item_name") or doc.get("item_name") or "",
                "expanded": True,
            })

            if len(expanded) >= MAX_PARENT_DOCS:
                break

        logger.info(
            f"父块扩展: {len(docs)} 个子块 -> {len(expanded)} 个文档，"
            f"命中父块 {len(seen_parents)} 个"
        )

        return {"expanded_docs": expanded}
