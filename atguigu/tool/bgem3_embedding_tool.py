# 获取到初始化的模型对象（单例）
import threading
from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from atguigu.config.config import BGEM3Config
from atguigu.tool.json_format_tool import json_format

embedding_model = None

# ============================================================================
# 【稳定性修复：BGE 单例加锁】
#
# 为什么需要：
# - 导入服务用 BackgroundTasks 可能同时跑多个文件；
# - 多个线程会同时执行 `if embedding_model is None`；
# - 两个线程可能各自加载一份 BGE，GPU/CPU 内存和 CUDA context 冲突，
#   在 Windows 上可能表现为 0xC0000005（Access Violation）原生崩溃；
# - PyTorch 模型也不保证同一模块被多个线程同时 forward 是安全的。
#
# 做法：
# - _model_lock：双重检查锁，保证只加载一次；
# - _encode_lock：同一时刻只允许一个线程做 encode，
#   避免并发 forward 导致显存/上下文问题。
# ============================================================================
_model_lock = threading.Lock()
_encode_lock = threading.Lock()


def get_embedding_model():
    """
    获取 BGE-M3 模型单例。

    双重检查：
    - 第一次检查避免每次调用都抢锁；
    - 第二次检查在锁内，防止两个线程同时加载。
    """
    global embedding_model

    if embedding_model is None:
        with _model_lock:
            if embedding_model is None:
                embedding_model = BGEM3EmbeddingFunction(
                    model_name=BGEM3Config.bge_m3_path,
                    device=BGEM3Config.bge_device,
                    use_fp16=BGEM3Config.bge_fp16,
                )

    return embedding_model


def get_bgem3_embeddings(text_list: List[str]):
    """
    把文本列表编码成 dense + sparse 向量。

    encode 阶段串行执行：
    - BGE 模型是进程内共享单例；
    - 并发 forward 可能造成显存/上下文问题；
    - 导入本身应该串行，不追求高并发。
    """
    embedding_model = get_embedding_model()

    with _encode_lock:
        result = embedding_model.encode_documents(text_list)

    return {
        "dense": [item.tolist() for item in result.get("dense")],
        "sparse": [dict(zip(item.indices.tolist(), item.data.tolist())) for item in result.get("sparse")],
    }


if __name__ == '__main__':
    res = get_bgem3_embeddings(["我爱杨幂","我爱赵丽颖"])
    print(json_format(res))





