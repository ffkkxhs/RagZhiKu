import asyncio
from asyncio import Queue

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Body, Query
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BodyPrams(BaseModel):
    session_id: str = Field(...,description="session_id")
    query: str = Field(...,description="query")

q_dict = {}
async def make_data(session_id,query):
    if session_id not in q_dict:
        q_dict[session_id]=Queue()

    q = q_dict[session_id]

    await asyncio.sleep(1)
    await q.put({"event":"progress","data":f"A{query}"})
    await q.put({"event":"progress","data":f"B{query}"})
    await asyncio.sleep(5)
    await q.put({"event":"progress","data":f"C"})
    await q.put({"event":"progress","data":f"D"})

    await q.put({"event":"final","data":f"EEEEEEEEEE"})


@app.post("/startwork")
def startwork(
    background_tasks: BackgroundTasks,
    body_prams: BodyPrams = Body(...,description="请求体参数")
):
    session_id = body_prams.session_id
    query = body_prams.query

    background_tasks.add_task(make_data,session_id,query)

    return {"msg":"开始造数据"}

async def generate_data(session_id):
    while session_id not in q_dict:
        await asyncio.sleep(1)

    q = q_dict[session_id]
    while True:
        try:
            data = await q.get()
            # a = 1 / 0
            await asyncio.sleep(1)
            yield f"event: {data.get('event')}\n"
            yield f"data: {data.get('data')}\n\n"
        except Exception as e:
            yield f"event: error\n"
            yield f"data: {e}\n\n"


@app.get("/sse02")
def sse02(
    session_id:str = Query(...,description="session_id"),
):
    return StreamingResponse(generate_data(session_id),media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)