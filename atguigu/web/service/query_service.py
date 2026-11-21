import json
import time
import uuid
from queue import Empty

import uvicorn
from fastapi import FastAPI, Body, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from atguigu.query_process.main_graph import QueryMainGraphRunner
from atguigu.tool.mongo_tool import clear_history, get_recent_messages
from atguigu.tool.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_q, put_data, get_task_info, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

app = FastAPI(
    title="掌柜智库-检索API",
    description="此文档是掌柜智库检索流程的API接口说明"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"msg": "ok"}


@app.delete("/history/{session_id}")
def delete_history(
    session_id: str,
):
    clear_history(session_id)
    return {"msg": "ok"}


@app.get("/history/{session_id}")
def get_history(
    session_id: str,
):
    history_list =  get_recent_messages(session_id)

    # 先把 ObjectId 转成字符串，再展开原文档；否则 **item 会把 _id 覆盖回 ObjectId，
    # FastAPI 序列化时会直接报 500。
    history_list = [
        {
            **item,
            "_id": str(item.get("_id")),
        }
        for item in history_list
    ]

    history_list.sort(key=lambda item: item["ts"])

    return {"items":history_list}



class queryBody(BaseModel):
    query: str = Field(...,description="query")
    session_id: str = Field(...,description="session_id")

@app.post("/query")
def query(
        querybody: queryBody = Body(...,description="query"),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    task_id = str(uuid.uuid4())

    # BackgroundTasks 实例不是可调用对象，必须用 add_task 注册后台函数
    background_tasks.add_task(run_main_graph, task_id, querybody.query, querybody.session_id)

    # 前端要拿 task_id 去连 /stream/{task_id}，这里必须返回
    return {"task_id": task_id}



def run_main_graph(task_id,query,session_id):

    # 查询图读取的字段名是 original_query，不是 query
    init_state={
        "task_id": task_id,
        "original_query": query,
        "session_id": session_id
    }

    try:
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        put_data(task_id,event="progress",data=get_task_info(task_id))
        QueryMainGraphRunner.create_and_run(init_state)
        update_task_status(task_id,TASK_STATUS_COMPLETED)
        put_data(task_id,event="progress",data=get_task_info(task_id))
    except:
        update_task_status(task_id,TASK_STATUS_FAILED)
        put_data(task_id,event="error",data=get_task_info(task_id))
        raise




def generate_data(task_id):
    """
    从任务队列里逐条取出事件，转成 SSE 格式。

    队列里的元素是 {"event": ..., "data": ...}，不是 dict 表本身；
    前端用 JSON.parse(e.data)，所以 data 必须 json.dumps。
    收到 final 或 error 后结束，避免无限循环。
    """
    q = get_q(task_id)

    while True:
        try:
            item = q.get(timeout=120)
        except Empty:
            yield 'event: error\ndata: {"error": "等待响应超时"}\n\n'
            return

        event = item.get("event", "message")
        data = json.dumps(item.get("data"), ensure_ascii=False, default=str)
        yield f"event: {event}\ndata: {data}\n\n"

        if event in ("final", "error"):
            return

@app.get("/stream/{task_id}")
def stream(
        task_id: str,
):
    return StreamingResponse(generate_data(task_id),media_type="text/event-stream")


if __name__ == "__main__":
    # chat.html 连的是 8001，这里给一个直接可跑的入口
    uvicorn.run(app, host="0.0.0.0", port=8001)



