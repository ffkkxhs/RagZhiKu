# atguigu/tool/llm_tool.py
"""
LLM 调用的统一重试封装。

为什么需要：
- 调用大模型走的是公网 HTTPS，偶尔会遇到连接重置、超时、429；
- 这些是瞬时错误，重试一次通常就好了；
- 如果每个节点都自己写 try/except，代码会重复，而且容易漏。

这个模块只做一件事：
把“可重试的错误”识别出来，按指数退避重试几次，最终仍失败才抛给上层。

不重试的错误：
- 401/403（Key 无效、没权限）
- 400（参数错误、模型名不存在）
- 这些重试多少次都没用，应该直接失败并暴露问题。
"""

import random
import time

from langchain_openai.chat_models.base import (
    OpenAIConnectionError,
    OpenAIRateLimitError,
    OpenAITimeoutError,
)
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from atguigu.tool.logger import logger

# 这些错误属于“现在不行，等一会儿可能行”，值得重试
RETRYABLE_ERRORS = (
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIRateLimitError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


def _retry_after_seconds(error) -> float | None:
    """
    读取服务端返回的 Retry-After 头。

    为什么需要：
    - 遇到 429 时，服务端可能明确告诉客户端“多少秒后再试”；
    - 直接按固定间隔重试，可能仍然被限流；
    - 有 Retry-After 就优先听服务端的。
    """
    response = getattr(error, "response", None)
    if response is None:
        return None

    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def invoke_with_retry(
    llm,
    messages,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
):
    """
    带指数退避的同步 LLM 调用。

    参数：
    - llm: langchain 的 chat model 对象
    - messages: 传给 llm.invoke 的消息（字符串或消息列表）
    - max_retries: 最多重试几次（总共尝试 max_retries + 1 次）
    - base_delay: 第一次重试等待的秒数
    - max_delay: 单次等待的上限

    返回：
    - llm.invoke 的原始返回值

    重试策略：
    - 第 1 次失败等 ~1s，第 2 次 ~2s，第 3 次 ~4s；
    - 每次乘一个 0.5~1.5 的随机抖动，避免多个任务同时重试；
    - 如果服务端给了 Retry-After，优先按它等。
    """
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)

        except RETRYABLE_ERRORS as error:
            # 最后一次还失败，就把原始异常抛出去
            if attempt == max_retries:
                raise

            retry_after = _retry_after_seconds(error)

            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(max_delay, base_delay * (2 ** attempt))
                # 加抖动：0.5~1.5 倍，避免所有请求同一时刻再次撞上去
                delay *= 0.5 + random.random()

            logger.warning(
                f"LLM 调用失败（第 {attempt + 1}/{max_retries + 1} 次），"
                f"{delay:.1f}s 后重试: {type(error).__name__}: {error}"
            )
            time.sleep(delay)
