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

         embeddings = get_bgem3_embeddings([rewritten_query])

         dense = embeddings.get("dense")[0]
         sparse = embeddings.get("sparse")[0]

         expr = f"item_name in {json.dumps(item_names)}"
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
             output_fields=["id", "item_name", "file_title", "content", "title"],
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