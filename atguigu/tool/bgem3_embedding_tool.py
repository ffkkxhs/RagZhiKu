# 获取到初始化的模型对象（单例）
from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from atguigu.config.config import BGEM3Config
from atguigu.tool.json_format_tool import json_format

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        #初始化模型
        embedding_model = BGEM3EmbeddingFunction(
            model_name=BGEM3Config.bge_m3_path,
            device=BGEM3Config.bge_device,
            use_fp16=BGEM3Config.bge_fp16,
        )
    return embedding_model


def get_bgem3_embeddings(text_list: List[str]):
    embedding_model = get_embedding_model()
    result = embedding_model.encode_documents(text_list)

    return {
        "dense": [item.tolist() for item in result.get("dense")],
        "sparse": [dict(zip(item.indices.tolist(),item.data.tolist())) for item in result.get("sparse")]
    }


if __name__ == '__main__':
    res = get_bgem3_embeddings(["我爱杨幂","我爱赵丽颖"])
    print(json_format(res))





