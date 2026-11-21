import uuid
import time
from datetime import datetime
import shutil
from pathlib import Path
import fastapi
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from atguigu.config.config import MinioConfig
from atguigu.import_process.main_graph import MainGraphRunner
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger
from atguigu.tool.minio_client_tool import get_minio_client
from atguigu.tool.task_utils import get_task_info, add_running_task, add_done_task, add_node_duration, \
    update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

app = FastAPI(
    title="掌柜智库-导入API",
    description="此文档是掌柜智库导入流程的API接口说明"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)


@app.post("/upload")
def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...,description="file to upload"),
):



    task_id = str(uuid.uuid4())

    add_running_task(task_id,"upload_file")
    start_time = time.time()


    file_dir_obj = Path(rf"D:\output\{datetime.now().strftime('%Y%m%d%H%M%S')}")
    if not file_dir_obj.exists():
        file_dir_obj.mkdir(parents=True, exist_ok=True)

    # 只取文件名，避免 filename 里带 ../ 之类的内容把文件写到目录外
    file_name = Path(file.filename or "upload.bin").name
    file_path_obj = file_dir_obj / file_name

    with open(file_path_obj,"wb")as f:
        shutil.copyfileobj(file.file, f, 1024*1024*2)



    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name=MinioConfig.minio_bucket_name,
        object_name=f"upload_file/{datetime.now().strftime('%Y%m%d')}/{file_name}",
        file_path=str(file_path_obj),
    )
    logger.info("上传文件结束")

    add_done_task(task_id,"upload_file")
    end_time = time.time()
    add_node_duration(task_id, "upload_file", end_time - start_time)


    background_tasks.add_task(run_main_graph, task_id, str(file_dir_obj), str(file_path_obj))
    return {"task_id": task_id}


def run_main_graph(task_id, file_dir_obj, file_path_obj):
    # 导入图读取的是 local_dir / local_file_path，字段名必须和节点入口一致
    init_state = {
        "task_id": task_id,
        "local_dir": str(file_dir_obj),
        "local_file_path": str(file_path_obj),
    }

    try:

        update_task_status(task_id, TASK_STATUS_PROCESSING)

        result = MainGraphRunner.create_and_run(init_state)
        logger.info(json_format(result))

        update_task_status(task_id, TASK_STATUS_COMPLETED)

    except:

        update_task_status(task_id, TASK_STATUS_FAILED)
        raise

#书写轮询接口返回实时状态数据
@app.get("/status/{task_id}")
def get_status(task_id: str = fastapi.Path(...,description="用户对应的任务id")):
    return get_task_info(task_id)


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=8000)  # port 必须是 int，字符串会导致 uvicorn 启动日志的 %d 格式化报错