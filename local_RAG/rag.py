# 필요한 라이브러리들을 불러옵니다
import torch  # PyTorch 라이브러리, 텐서 연산을 위해 사용
import ollama  # Ollama API 사용을 위한 라이브러리
import os  # 운영 체제와 관련된 기능을 사용하기 위한 라이브러리
from openai import OpenAI  # OpenAI API 클라이언트
import argparse  # 명령줄 인자를 파싱하기 위한 라이브러리
import json  # JSON 데이터를 다루기 위한 라이브러리

# 콘솔 출력 시 사용할 색상 코드를 정의합니다
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# 파일을 열어 내용을 읽는 함수
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# 사용자 입력과 관련된 컨텍스트를 가져오는 함수
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # vault가 비어있는지 확인
        return []
    # 사용자 입력에 대한 임베딩을 생성
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # 입력 임베딩과 vault 임베딩 간의 코사인 유사도를 계산
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))  # top_k가 vault 크기를 초과하지 않도록 조정
    # 가장 유사한 top_k개의 컨텍스트 인덱스를 가져옴
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # 관련 컨텍스트를 반환
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# 사용자 쿼리를 재작성하는 함수
def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    # 최근 대화 기록을 문자열로 만듦
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    # 쿼리 재작성을 위한 프롬프트
    prompt = f"""다음 쿼리를 대화 기록의 관련 컨텍스트를 통합하여 다시 작성하세요.
    다시 작성된 쿼리는:
    
    - 원래 쿼리의 핵심 의도와 의미를 유지해야 합니다
    - 관련 컨텍스트를 검색하는 데 더 구체적이고 정보가 풍부하도록 쿼리를 확장하고 명확히 해야 합니다
    - 원래 쿼리에서 벗어나는 새로운 주제나 쿼리를 도입하지 않아야 합니다
    - 절대로 원래 쿼리에 답하지 말고, 대신 새로운 쿼리로 다시 작성하고 확장하는 데 집중하세요
    
    다시 작성된 쿼리 텍스트만 반환하고, 추가 형식이나 설명은 포함하지 마세요.
    
    대화 기록:
    {context}
    
    원래 쿼리: [{user_input}]
    
    다시 작성된 쿼리: 
    """
    # Ollama API를 사용하여 쿼리 재작성
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

# Ollama 채팅 기능을 구현하는 함수 (이어서)
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    # 사용자 입력을 대화 기록에 추가
    conversation_history.append({"role": "user", "content": user_input})
    
    # 대화 기록이 있으면 쿼리를 재작성
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "원래 쿼리: " + user_input + RESET_COLOR)
        print(PINK + "다시 작성된 쿼리: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    # 관련 컨텍스트 가져오기
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("문서에서 추출된 컨텍스트: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "관련 컨텍스트를 찾을 수 없습니다." + RESET_COLOR)
    
    # 사용자 입력에 컨텍스트 추가
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\n관련 컨텍스트:\n" + context_str
    
    # 대화 기록 업데이트
    conversation_history[-1]["content"] = user_input_with_context
    
    # Ollama API에 보낼 메시지 구성
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    # Ollama API를 사용하여 응답 생성
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    # 생성된 응답을 대화 기록에 추가
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# 메인 프로그램 시작
print(NEON_GREEN + "명령줄 인수를 파싱 중..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama 채팅")
parser.add_argument("--model", default="llama3.2:3b", help="사용할 Ollama 모델 (기본값: llama3.2:3b)")
args = parser.parse_args()

print(NEON_GREEN + "Ollama API 클라이언트 초기화 중..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3.2:3b'
)

print(NEON_GREEN + "vault 내용 로드 중..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

print(NEON_GREEN + "vault 내용에 대한 임베딩 생성 중..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

print("임베딩을 텐서로 변환 중...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("vault의 각 라인에 대한 임베딩:")
print(vault_embeddings_tensor)

print("대화 루프 시작...")
conversation_history = []
system_message = "당신은 주어진 텍스트에서 가장 유용한 정보를 추출하는 데 전문가인 도움이 되는 어시스턴트입니다. 또한 주어진 컨텍스트 외에도 사용자 쿼리와 관련된 추가 정보를 제공하세요."

# 메인 대화 루프
while True:
    user_input = input(YELLOW + "문서에 대해 질문해주세요 ('종료'를 입력하면 프로그램이 종료됩니다): " + RESET_COLOR)
    if user_input.lower() == '종료':
        break
    
    # Ollama 채팅 함수를 호출하여 응답 생성
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "응답: \n\n" + response + RESET_COLOR)

print("대화를 종료합니다. 감사합니다!")