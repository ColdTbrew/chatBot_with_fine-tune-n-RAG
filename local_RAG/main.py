from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import torch
import ollama
import json
import re
import fitz  # PyMuPDF
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FastAPI 애플리케이션 초기화
app = FastAPI(title="LLM RAG API", description="파일 업로드 및 RAG 기반 문서 질의 응답 API", version="1.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "uploads/processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 콘솔 색상 정의
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# 파일 저장 비동기 함수
async def save_uploaded_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        buffer.write(await upload_file.read())

# 텍스트를 청크 단위로 나누는 함수
def split_text_into_chunks(text, chunk_size=1000):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < chunk_size:
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# PDF 파일 처리
def process_pdf(file_path, filename):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    chunks = split_text_into_chunks(text)
    
    vault_path = f"{PROCESSED_DIR}/{filename}_vault.txt"
    with open(vault_path, "w", encoding="utf-8") as vault_file:
        for chunk in chunks:
            vault_file.write(chunk.strip() + "\n")
    
    return {"message": "PDF 파일이 성공적으로 처리되었습니다."}

# PDF 파일 업로드 API
@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")
    
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    await save_uploaded_file(file, file_path)
    return process_pdf(file_path, file.filename.split('.')[0])

# JSON 파일 처리
def process_json(file_path, filename):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    text = json.dumps(data, ensure_ascii=False, indent=4)
    chunks = split_text_into_chunks(text)
    
    vault_path = f"{PROCESSED_DIR}/{filename}_vault.txt"
    with open(vault_path, "w", encoding="utf-8") as vault_file:
        for chunk in chunks:
            vault_file.write(chunk.strip() + "\n")
    
    return {"message": "JSON 파일이 성공적으로 처리되었습니다."}

# JSON 파일 업로드 API
@app.post("/upload/json/")
async def upload_json(file: UploadFile = File(...)):
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="JSON 파일만 업로드할 수 있습니다.")
    
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    await save_uploaded_file(file, file_path)
    return process_json(file_path, file.filename.split('.')[0])

# 파일에서 텍스트를 추출하여 벡터화
@app.post("/upload/text/")
async def upload_text(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="텍스트 파일만 업로드할 수 있습니다.")
    
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    await save_uploaded_file(file, file_path)
    
    with open(file_path, 'r', encoding="utf-8") as txt_file:
        text = txt_file.read()
    
    chunks = split_text_into_chunks(text)
    vault_path = f"{PROCESSED_DIR}/{file.filename}_vault.txt"
    with open(vault_path, "w", encoding="utf-8") as vault_file:
        for chunk in chunks:
            vault_file.write(chunk.strip() + "\n")
    
    return {"message": "텍스트 파일이 성공적으로 처리되었습니다."}

# RAG 관련 함수 정의
vault_content = []
vault_embeddings = []

@app.get("/load_vault/")
def load_vault():
    global vault_content, vault_embeddings
    
    if os.path.exists("uploads/processed"):
        vault_content = []
        vault_embeddings = []
        for filename in os.listdir("uploads/processed"):
            if filename.endswith("_vault.txt"):
                with open(f"uploads/processed/{filename}", "r", encoding='utf-8') as file:
                    content = file.readlines()
                    vault_content.extend(content)
                    
        for content in vault_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])
        
    return {"message": "Vault 데이터가 로드되었습니다.", "entries": len(vault_content)}

# 유사한 컨텍스트 검색 함수
def get_relevant_context(query, top_k=3):
    if not vault_embeddings:
        return []
    
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=query)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), torch.tensor(vault_embeddings))
    top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1].tolist()
    return [vault_content[idx].strip() for idx in top_indices]

# Ollama 기반 RAG 질의응답 API
@app.post("/chat/")
def query_llm(message: Message):
    prompt = message.message
    relevant_context = get_relevant_context(prompt)
    context_str = "\n".join(relevant_context) if relevant_context else ""
    
    messages = [
        {"role": "system", "content": "당신은 mbti 정보를 알려주는 챗봇입니다. 한국어로 대답해줘"},
        {"role": "user", "content": prompt + "\n\n" + context_str}
    ]
    print(messages)
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='llama3.2-mbti-v1.0:latest')
    response = client.chat.completions.create(
        model='llama3.2-mbti-v1.0:latest',
        messages=messages,
        max_tokens=2000,
    )
    
    # return {"response": response.choices[0].message.content}
    return {"reply": response.choices[0].message.content}
