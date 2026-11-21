import time

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from atguigu.config.config import MilvusConfig
from atguigu.tool.logger import logger

milvus_client = None


def get_milvus_client(max_retries: int = 3, retry_interval: float = 1.0):
    """
    获取 Milvus 客户端单例，并在创建时做一次带重试的连通性预检。

    为什么需要预检：
    pymilvus 只有在第一次真正操作时才会暴露连接失败，报错是
    "illegal connection params or server unavailable"，很难判断是配置错还是服务没起。
    这里在创建客户端后立刻 list_collections()，失败就重试几次，
    最终抛一个写清楚地址和排查方向的 RuntimeError。

    参数：
    - max_retries: 最多尝试几次
    - retry_interval: 每次重试前等待的秒数（按次数递增）
    """
    global milvus_client

    if milvus_client is not None:
        return milvus_client

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            client = MilvusClient(
                uri=MilvusConfig.milvus_url,
                timeout=5,
            )
            # 真正触发一次 RPC，确认服务可达
            client.list_collections()
            milvus_client = client
            return milvus_client
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = retry_interval * attempt
                logger.warning(
                    f"Milvus 连接失败（第 {attempt}/{max_retries} 次），"
                    f"{wait}s 后重试: {e}"
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Milvus 不可达: {MilvusConfig.milvus_url}。"
        f"请确认 Milvus 的 19530 端口已启动（docker compose up -d etcd minio standalone），"
        f"并检查防火墙/端口映射。原始错误: {last_error}"
    ) from last_error



def get_req_list(
    dense_data,
    sparse_data,
    dense_anns_field,
    sparse_anns_field,
    dense_param = None,
    sparse_param = None,
    limit = 10,
    expr = None
):

    if dense_param is None:
        dense_param = {
            "metric_type": "COSINE",
        }

    if sparse_param is None:
        sparse_param = {
            "metric_type": "IP",
        }

    dense_req = AnnSearchRequest(
        data=[dense_data],
        param=dense_param,
        anns_field=dense_anns_field,
        limit=limit,
        expr=expr
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_data],
        param=sparse_param,
        anns_field=sparse_anns_field,
        limit=limit,
        expr=expr
    )

    return [dense_req, sparse_req]


def hybrid_search(
    reqs,
    collection_name,
    output_fields,
    limit = 10,
    ranker = (0.5,0.5)
):
    milvus_client = get_milvus_client()
    ranker = WeightedRanker(*ranker)

    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        limit=limit,
        ranker=ranker,
        reqs=reqs,
        output_fields=output_fields,
    )

    return res

