# atguigu/query_process/nodes/node_rrf.py

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger

class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    name: str = "node_rrf"

    def process(self, state: QueryGraphState):
        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")

        weight_chunks = [
            (embedding_chunks,1),
            (hyde_embedding_chunks,1)
        ]

        rrf_dict = {}
        for chunks,weight in weight_chunks:
            for idx,chunk in enumerate(chunks,start=1):
                id = chunk.get("id")
                score = weight / (idx + 60)
                if id in rrf_dict:

                    rrf_dict[id]['score'] += score
                else:
                    chunk["score"] = score
                    rrf_dict[id] = chunk


        rrf_chunks = sorted(rrf_dict.values(),key=lambda x:x["score"],reverse=True)[:10]
        return {
            "rrf_chunks":rrf_chunks
        }