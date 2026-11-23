# atguigu/query_process/nodes/node_search_embedding.py
import json

from atguigu.config.config import MilvusConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import get_req_list, hybrid_search


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: QueryGraphState):


         rewritten_query = state.get("rewritten_query")
         item_names = state.get("item_names")
         strict_filter = state.get("strict_filter", True)

         embeddings = get_bgem3_embeddings([rewritten_query])

         dense = embeddings.get("dense")[0]
         sparse = embeddings.get("sparse")[0]

         # ================================================================
         # 【改造说明：硬过滤改成受 strict_filter 控制的软过滤】
         #
         # 原逻辑只要 item_names 非空就拼 `item_name in [...]`。
         # 一旦主体名识别错了，正确文档会被直接排除，后面再好的 rerank 也救不回来。
         #
         # 新逻辑：
         # - 主体名确认得很确定（strict_filter=True）才做硬过滤；
         # - 只是 0.6~0.8 的候选（strict_filter=False）就不加过滤，
         #   让 dense/sparse 先召回更多候选，交给 rerank 判断。
         # 这是通用策略，不依赖具体领域。
         # ================================================================
         expr = (
             f"item_name in {json.dumps(item_names)}"
             if strict_filter and item_names
             else None
         )
         reqs = get_req_list(
             dense_data=dense,
             sparse_data=sparse,
             dense_anns_field="dense_vector",
             sparse_anns_field="sparse_vector",
             expr=expr
         )

         res = hybrid_search(
             collection_name=MilvusConfig.chunks_collection,
             reqs=reqs,
             # 新字段一起带出来，rerank/答案节点才能拿到完整标题路径和元数据
             output_fields=[
                 "id", "item_name", "file_title", "content", "title",
                 "title_path", "level", "parent_id", "section_id",
                 "chunk_id", "metadata",
             ],
             ranker=(0.6, 0.4)
         )

         chunks = [
             {
                 **item.get("entity"),
                 "score": item.get("distance"),
                 "source": "local"
             }
             for item in res[0]
         ]

         return {
             "embedding_chunks": chunks
         }



         # return state
         return {"embedding_chunks":  []}