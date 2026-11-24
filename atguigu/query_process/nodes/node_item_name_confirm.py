# atguigu/query_process/nodes/node_item_name_confirm.py

import json

from langchain.chat_models import init_chat_model

from atguigu.config.config import LLMConfig, MilvusConfig
from atguigu.config.prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.llm_tool import invoke_with_retry
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import get_milvus_client, get_req_list, hybrid_search
from atguigu.tool.mongo_tool import get_recent_messages, save_chat_message, update_history_item_names, clear_history


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState):

        session_id = state.get("session_id")
        original_query = state.get("original_query")

        if not original_query:
            logger.error("query is empty")
            raise Exception("query is empty")

        if not session_id:
            logger.error("session_id is empty")
            raise Exception("session_id is empty")


        history_list = get_recent_messages(session_id)
        id = save_chat_message(session_id=session_id,text=original_query,role="user")

        history_list_str = ""

        # Mongo 按 ts 倒序返回，拼 prompt 前转回时间正序
        for history_message in reversed(history_list):
            history_list_str += f"{history_message.get('role')}:{history_message.get('text')}\n\n"

        #todo
        # return history_list_str,session_id,original_query


        message = [
            {"role":"system","content":ITEM_NAME_EXTRACT_SYSTEM_PROMPT},
            {"role":"user","content":ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_list_str,
                original_query=original_query,
            )},
        ]

        llm = init_chat_model(
            model_provider="openai",
            model=LLMConfig.llm_default_model,
            api_key=LLMConfig.openai_api_key,
            base_url=LLMConfig.openai_api_base,
            temperature=LLMConfig.llm_default_temperature,
            # 单次请求超时，避免网络异常时一直挂着
            timeout=60,
        )

        # 意图提取也是公网调用，偶发连接问题同样需要重试
        res = invoke_with_retry(llm, message)

        res_json = res.content
        res_json = res_json.replace("```json", "").replace("```", "")
        res_data = json.loads(res_json)

        item_names = res_data.get("item_names",[])
        # 旅游领域类别：景点/交通/美食/住宿/线路/综合；空字符串表示不明确
        query_category = str(res_data.get("category") or "").strip()
        if query_category not in {"景点", "交通", "美食", "住宿", "线路", "综合"}:
            query_category = ""
        rewritten_query =res_data.get("rewritten_query","")

        #todo
        # return item_names, rewritten_query

        search_item_names = []
        final_item_names = []
        answer = ""
        # 默认严格过滤；只有在主体名不确定时才放宽
        strict_filter = True

        if item_names:
            embeddings =get_bgem3_embeddings(item_names)

            for idx,item_name in enumerate(item_names):
                # ============================================================
                # 【改造说明：先做精确/别名匹配，再做向量兜底】
                #
                # 原逻辑完全依赖向量相似度。城市名之间天然相似
                # （成都/杭州/成都美食），分数容易都落在 0.6~0.8，
                # 结果全部变成“待选词”，查询被短路。
                #
                # 新逻辑：
                # 1. 先在 kb_item_names 里做精确匹配；命中就直接确认，不看分数；
                # 2. 没有再走向量检索，作为模糊兜底。
                # 这是通用的实体解析顺序：精确 > 别名 > 向量。
                # ============================================================
                exact_rows = self._exact_item_lookup(item_name)
                if exact_rows:
                    for row in exact_rows:
                        search_item_names.append({
                            "score": 1.0,
                            "query_item_name": item_name,
                            "search_item_name": row.get("item_name"),
                        })
                    continue

                reqs = get_req_list(
                    dense_data = embeddings.get("dense")[idx],
                    sparse_data = embeddings.get("sparse")[idx],
                    dense_anns_field = "dense_vector",
                    sparse_anns_field = "sparse_vector",
                )

                res = hybrid_search(
                    reqs=reqs,
                    collection_name=MilvusConfig.item_name_collection,
                    output_fields=["item_name", "file_title", "labels"],
                    limit=10,
                    ranker=(0.7, 0.3)
                )

                for i in res[0]:
                    search_item_names.append({
                        "score":i.get("distance"),
                        "query_item_name": item_name,
                        "search_item_name":i.get("entity").get("item_name")
                    })

            #todo
            # return final_item_names

            confirm_names = []
            option_names = []

            for search_item_name in search_item_names:
                # search_item_name.get("query_item_name")
                name = search_item_name.get("search_item_name")
                score = search_item_name.get("score")

                if score >= 0.8:
                    confirm_names.append(name)
                elif score < 0.8 and score > 0.6:
                    option_names.append(name)


            # 同一个商品可能被多个 query 命中，去重并保持顺序
            confirm_names = list(dict.fromkeys(confirm_names))
            option_names = list(dict.fromkeys(option_names))

            if confirm_names:
                final_item_names = confirm_names
                answer = ""
                strict_filter = True
            elif option_names:
                # ============================================================
                # 【改造说明：0.6~0.8 不再短路，改成“带候选继续检索”】
                #
                # 原逻辑返回“待选词”后直接结束，用户没有得到答案，
                # 检索指标也全部是 0。
                #
                # 现在：
                # - 取前 3 个候选作为临时主体名；
                # - strict_filter=False，检索时不做 item_name 硬过滤；
                # - 让 dense/sparse/rerank 去判断哪个候选真正相关。
                # 这样即使主体名不确定，也还有机会召回正确文档。
                #
                # 后续如果要做“让用户选”的交互，可以把候选放进
                # clarification_options 字段，由前端展示，而不是终止流程。
                # ============================================================
                final_item_names = option_names[:3]
                strict_filter = False
                answer = ""
                logger.warning(
                    f"主体名不确定，使用候选继续检索: {final_item_names}"
                )
            else:
                final_item_names = []
                answer = "不好意思，没有识别到您问的商品，请重新提问"
                strict_filter = True

            #todo
            # return final_item_names,answer


        else:   #  item_name为空时的兜底
            answer = "没有识别到您问的商品，请补充商品型号后重试"
            final_item_names = []
            strict_filter = True

        if answer:
            message_id = save_chat_message(session_id, "assistant", answer)
            
        update_history_item_names([id], item_names=final_item_names, rewritten_query=rewritten_query)


        return {
            "original_query": original_query,
            "answer": answer,
            "item_names": final_item_names,
            # 查询类别只作为软加权信号，不单独做硬过滤
            "query_category": query_category,
            "rewritten_query": rewritten_query,
            "history": get_recent_messages(session_id),
            "session_id": session_id,
            # 检索节点根据这个开关决定要不要做 item_name 硬过滤
            "strict_filter": strict_filter,
        }

    def _exact_item_lookup(self, item_name):
        """
        在 kb_item_names 里做精确匹配。

        为什么需要：
        - 向量相似度适合“模糊找候选”，不适合“确认同一个实体”；
        - 城市名、书名、课程名这类标准名称，精确匹配更可靠；
        - 精确命中时直接确认，可以避免“成都/杭州/成都美食”分数接近
          导致整条查询变成“待选词”。

        参数：
        - item_name: LLM 提取出的主体名

        返回：
        - list[dict]：命中的行；没有命中或集合不存在时返回空列表
        """
        try:
            client = get_milvus_client()
            collection = MilvusConfig.item_name_collection

            if not client.has_collection(collection):
                return []

            client.load_collection(collection)

            # 用 json.dumps 生成带双引号的字符串字面量
            filter_expr = f"item_name == {json.dumps(item_name, ensure_ascii=False)}"

            return client.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=["item_name", "file_title", "labels"],
                limit=10,
            )

        except Exception as error:
            # 精确匹配失败不应该阻断查询，回退到向量检索
            logger.warning(f"主体名精确匹配失败，回退向量检索: {error}")
            return []


if __name__ == '__main__':
    session_id = "test_001"

    # save_chat_message(session_id, "user", "咨询下烫金机。")
    # save_chat_message(session_id, "assistant", "您好。请问是哪个型号")
    # save_chat_message(session_id, "user", "hak180")
    # save_chat_message(session_id, "assistant", "具体有什么问题呢？")

    init_state = {
        "session_id": "test_001",
        "original_query": "咨询下烫金机？"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    logger.info(json_format(result))




