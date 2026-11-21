import time
from queue import Queue

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Query
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

q_dict={}
def make_email(session_id):
    if session_id not in q_dict:
        q_dict[session_id]=Queue()

    q_dict[session_id].put("A")
    q_dict[session_id].put("B")
    q_dict[session_id].put("C")
    time.sleep(5)
    q_dict[session_id].put("D")
    q_dict[session_id].put("E")

    q_dict[session_id].put(None)



@app.get("/startwork/{session_id}")
def startwork(
    background_tasks: BackgroundTasks,
    session_id: str ,
):
    background_tasks.add_task(make_email,session_id=session_id)
    return {"msg": "making emails"}


def generate_data(session_id:str):

    while session_id not in q_dict:
        time.sleep(1)


    while True:
        time.sleep(1)
        email = q_dict[session_id].get()
        yield f"data: {email}\n\n"
        if email is None:
            break

@app.get("/sse01")
def sse01(
    session_id: str = Query(...,description="session_id"),
):
    return StreamingResponse(generate_data(session_id),media_type="text/event-stream")   # todo **********重点**********




if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)