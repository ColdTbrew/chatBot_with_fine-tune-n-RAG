from fastapi import FastAPI, Form
from pydantic import BaseModel
import requests
import json

app = FastAPI()

class RequestModel(BaseModel):
    model: str
    prompt: str

@app.post("/llama")
async def llama(prompt: str = Form(...)):
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
    # POST 요청 보내기
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        # 개별 JSON 객체로 분할, 이후 부터 추가된 코드
        json_objects = response.content.decode().strip().split("\n")
        # 각 JSON 객체를 Python 사전으로 변환
        data = [json.loads(obj) for obj in json_objects]
        res_text = ''
        # 변환된 데이터 출력
        for item in data:
            print(item)
            res_text += item['response']
        # return response.content.decode()
    else:
        res_text = "Error Code : " + response.status_code
    
    return res_text