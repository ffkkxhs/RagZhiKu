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

         embeddings = get_bgem3_embeddings([rewritten_query])

         dense = embeddings.get("dense")[0]
         sparse = embeddings.get("sparse")[0]

         # ================================================================
         # 【改造说明：默认不做 item_name 硬过滤】
         #
         # 之前只要 item_names 非空就拼 `item_name in [...]`。
         # 一旦主体名提取多了、写法不统一、或把多地名串成一个，
         # 正确文档会被直接排除，rerank 也救不回来。
         #
         # 现在：
         # - 检索以 rewritten_query 为准，召回所有文档；
         # - entity/category 在 rerank 阶段做软加权，不使用 Milvus 硬过滤；
         # - 只有用户显式选择某个分面时，未来才在 API 层加硬过滤。
         # ================================================================
         reqs = get_req_list(
             dense_data=dense,
             sparse_data=sparse,
             dense_anns_field="dense_vector",
             sparse_anns_field="sparse_vector",
         )

         res = hybrid_search(
             collection_name=MilvusConfig.chunks_collection,
             reqs=reqs,
             # 新字段一起带出来，rerank/答案节点才能拿到完整标题路径和旅游标签
             output_fields=[
                 "id", "item_name", "file_title", "content", "title",
                 "title_path", "level", "parent_id", "section_id",
                 "chunk_id", "labels",
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