# atguigu/import_process/nodes/node_bge_embedding.py
import json
from pathlib import Path

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger


class NodeBGEEmbedding(NodeBase):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):


        chunks = state.get("chunks")

        for i in range(0,len(chunks),3):
            temp_list = chunks[i:i + 3]

            # ================================================================
            # 【改造说明：embedding 输入加入旅游类别】
            # 原来只拼 item_name + content；现在把 category 也拼进去，
            # 让“交通/美食/住宿/线路/景点”在向量空间里更可区分。
            # content 本身已经带 title_path 前缀。
            # 这是 schema/标签改造的一部分，需要重新向量化。
            # ================================================================
            chunks_str_list = []
            for item in temp_list:
                labels = item.get("labels") or {}
                entity = item.get("item_name") or ""
                category = labels.get("category") or ""
                chunks_str_list.append(f"{entity}\n{category}\n{item.get('content')}")

            vec_list = get_bgem3_embeddings(chunks_str_list)   # todo     +++向量化核心+++

            dense_list = vec_list.get("dense")
            sparse_list = vec_list.get("sparse")

            for i,chunk in enumerate(temp_list):
                chunk["dense_vector"] = dense_list[i]
                chunk["sparse_vector"] = sparse_list[i]

        # 调试产物写到当前文档所在目录，不再写死到 hak180 目录
        md_path = state.get("md_path")
        out_dir = Path(md_path).parent if md_path else Path(state.get("local_dir") or ".")
        out_path = out_dir / "embedding_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)

        return {
            "chunks": chunks,
        }

if __name__ == '__main__':
    node = NodeBGEEmbedding()
    with open(r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\item_name_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    init_state = {
        "chunks": chunks,
        "md_path": r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册_new.md",
    }
    result = node(init_state)
    logger.info(json_format(result))

