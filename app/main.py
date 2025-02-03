from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestModel(BaseModel):
    model: str
    prompt: str

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: Message):
    prompt = message.message
    # 요청할 URL
    url = "http://host.docker.internal:11434/api/generate"
    # 요청에 포함할 데이터
    data = {
        "model": "llama3.2:3b",
        "prompt": prompt
    }
    # 요청 헤더
    headers = {
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # 개별 JSON 객체로 분할
        json_objects = response.content.decode().strip().split("\n")
        # 각 JSON 객체를 Python 사전으로 변환
        data = [json.loads(obj) for obj in json_objects]
        res_text = ''
        # 변환된 데이터 출력
        for item in data:
            print(item)
            res_text += item['response']
    else:
        res_text = f"Error Code : {response.status_code}"
    return {"reply": res_text}