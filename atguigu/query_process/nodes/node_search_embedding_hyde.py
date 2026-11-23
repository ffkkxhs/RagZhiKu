# atguigu/query_process/nodes/node_search_embedding_hyde.py
import json

from langchain.chat_models import init_chat_model

from atguigu.config.config import LLMConfig, MilvusConfig
from atguigu.config.prompt import HYDE_PROMPT
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.llm_tool import invoke_with_retry
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import get_req_list, hybrid_search


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        # 与普通检索保持一致：主体名不确定时不硬过滤
        strict_filter = state.get("strict_filter", True)

        llm = init_chat_model(
            model=LLMConfig.llm_default_model,
            model_provider="openai",
            base_url=LLMConfig.openai_api_base,
            api_key=LLMConfig.openai_api_key,
            temperature=LLMConfig.llm_default_temperature,
            # 单次请求超时，避免网络异常时一直挂着
            timeout=60,
        )

        messages = [
            {"role": "user", "content": HYDE_PROMPT.format(rewritten_query=rewritten_query)}
        ]

        # HyDE 也要调公网 LLM，同样走带重试的封装
        res = invoke_with_retry(llm, messages)
        hyde_answer = res.content

        query = rewritten_query + "\n" + hyde_answer


        embeddings = get_bgem3_embeddings([query])
        dense = embeddings.get("dense")[0]
        sparse = embeddings.get("sparse")[0]

        # 只有 strict_filter=True 才做硬过滤；否则放宽召回，交给 rerank
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
            output_fields=[
                "id", "item_name", "file_title", "content", "title",
                "title_path", "level", "parent_id", "section_id",
                "chunk_id", "metadata",
            ],
            ranker=(0.7, 0.3)
        )
        # print(json_format(res))

        chunks = [
            {
                **item.get("entity"),
                "score": item.get("distance"),
                "source": "local"
            }
            for item in res[0]
        ]

        return {
            "hyde_embedding_chunks": chunks
        }