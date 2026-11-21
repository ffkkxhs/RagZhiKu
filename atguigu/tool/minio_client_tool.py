
"""
@aaa:bbb
@ccc:ddd
@name: 
"""
import email.utils
import json
import os
from datetime import datetime, timedelta, timezone

import requests
from minio import Minio
from minio import time as minio_time

from atguigu.config.config import MinioConfig
from atguigu.tool.logger import logger

minio_client = None

# ---------------------------------------------------------------------------
# 时钟偏移补偿
# S3 SigV4 签名允许客户端与服务端时差 ±15 分钟，超出即报
#   S3 operation failed; code: RequestTimeTooSkewed
# 常见于 MinIO 跑在被挂起过的虚拟机/容器里（时钟落后但仍在走）。
# 服务端时钟改不了时，这里探测一次服务端 Date 头，把 minio 的签名时钟对齐过去。
# ---------------------------------------------------------------------------
_clock_offset = 0.0
_original_utcnow = minio_time.utcnow


def _patch_minio_clock(offset_seconds: float):
    """把 minio 内部签名用的 utcnow 整体平移 offset_seconds 秒"""
    minio_time.utcnow = lambda: _original_utcnow() + timedelta(
        seconds=offset_seconds
    )


def _probe_server_clock(endpoint: str, timeout: float = 5.0):
    """探测服务端时钟与本地时钟的差值（秒）"""
    url = f"http://{endpoint}/minio/health/live"
    resp = requests.get(url, timeout=timeout)
    date_header = resp.headers.get("Date")
    if not date_header:
        return None
    server_dt = email.utils.parsedate_to_datetime(date_header)
    if server_dt.tzinfo is None:
        server_dt = server_dt.replace(tzinfo=timezone.utc)
    return (server_dt - datetime.now(timezone.utc)).total_seconds()


def calibrate_clock(endpoint: str):
    """校准 minio 签名时钟；MINIO_TIME_OFFSET(秒) 可手工覆盖"""
    global _clock_offset

    env_offset = os.getenv("MINIO_TIME_OFFSET")
    if env_offset not in (None, ""):
        _clock_offset = float(env_offset)
        source = "环境变量 MINIO_TIME_OFFSET"
    else:
        probed = _probe_server_clock(endpoint)
        if probed is None:
            logger.warning("探测不到 MinIO 服务端 Date 头，跳过时钟校准")
            return 0.0
        _clock_offset = probed
        source = "服务端 Date 头探测"

    _patch_minio_clock(_clock_offset)

    if abs(_clock_offset) >= 60:
        logger.warning(
            f"MinIO 服务端时钟偏移 {_clock_offset:+.0f}s "
            f"({_clock_offset / 3600:+.2f}h，来源:{source})，已对签名时钟做补偿"
        )
    else:
        logger.info(f"MinIO 时钟正常（偏移 {_clock_offset:+.1f}s，来源:{source}）")

    return _clock_offset


def get_minio_client():
    global minio_client

    if minio_client is None:
        minio_client = Minio(
            endpoint=MinioConfig.minio_endpoint,
            access_key=MinioConfig.minio_access_key,
            secret_key=MinioConfig.minio_secret_key,
            secure=False,  # 用http
        )
        bucket_name = MinioConfig.minio_bucket_name

        calibrate_clock(MinioConfig.minio_endpoint)

        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                    "Resource": f"arn:aws:s3:::{bucket_name}",
                },
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket_name}/*",
                },
            ],
        }

        minio_client.set_bucket_policy(bucket_name, json.dumps(policy))

    return minio_client


if __name__ == "__main__":
    client = get_minio_client()
    bucket = MinioConfig.minio_bucket_name

    # 冒烟测试：上传一个小对象 + 列目录 + 删除
    from io import BytesIO

    client.put_object(bucket, "_clock_probe.txt", BytesIO(b"ok"), length=2)
    logger.info("put_object 成功（签名时钟已对齐）")

    for obj in client.list_objects(bucket, prefix="_clock", recursive=True):
        logger.info(f"list_objects: {obj.object_name} {obj.size} bytes")

    client.remove_object(bucket, "_clock_probe.txt")
    logger.info("remove_object 成功")
