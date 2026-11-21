# atguigu/query_process/nodes/node_item_name_confirm.py

import json

from langchain.chat_models import init_chat_model

from atguigu.config.config import LLMConfig, MilvusConfig
from atguigu.config.prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import get_req_list, hybrid_search
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
            temperature=LLMConfig.llm_default_temperature
        )


        res = llm.invoke(message)

        res_json = res.content
        res_json = res_json.replace("```json", "").replace("```", "")
        res_data = json.loads(res_json)

        item_names = res_data.get("item_names",[])
        rewritten_query =res_data.get("rewritten_query","")

        #todo
        # return item_names, rewritten_query

        search_item_names = []
        final_item_names = []
        answer = ""
        
        if item_names:
            embeddings =get_bgem3_embeddings(item_names)

            for idx,item_name in enumerate(item_names):

                reqs = get_req_list(
                    dense_data = embeddings.get("dense")[idx],
                    sparse_data = embeddings.get("sparse")[idx],
                    dense_anns_field = "dense_vector",
                    sparse_anns_field = "sparse_vector",
                )

                res = hybrid_search(
                    reqs=reqs,
                    collection_name=MilvusConfig.item_name_collection,
                    output_fields=["item_name", "file_title"],        #  todo
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
            elif option_names:
                final_item_names = []
                answer = f"待选词(0.6-0.8)：{' '.join(option_names)}"
            else:
                final_item_names = []
                answer = "不好意思，没有识别到您问的商品，请重新提问"

            #todo
            # return final_item_names,answer


        else:   #  item_name为空时的兜底
            answer = "没有识别到您问的商品，请补充商品型号后重试"
            final_item_names = []
            
        if answer:
            message_id = save_chat_message(session_id, "assistant", answer)
            
        update_history_item_names([id], item_names=final_item_names, rewritten_query=rewritten_query)


        return {
            "original_query": original_query,
            "answer": answer,
            "item_names": final_item_names,
            "rewritten_query": rewritten_query,
            "history": get_recent_messages(session_id),
            "session_id": session_id
        }

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




