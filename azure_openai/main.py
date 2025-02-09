from openai import AzureOpenAI


azure_oai_deployment = "o1-mini" # 모델 이름
input_text = "write a Python program that shows a ball bouncing inside a spinning hexagon. The ball should be affected by gravity and friction, and it must bounce off the rotating walls realistically" # 입력 문장

client = AzureOpenAI(
        azure_endpoint = azure_oai_endpoint, 
        api_key=azure_oai_key,  
        api_version="2024-08-01-preview" # 2024-08-01-preview
)

response = client.chat.completions.create(
    model=azure_oai_deployment,
    messages=[
        {"role": "user", "content": input_text}
    ]
)
generated_text = response.choices[0].message.content

print("Response: " + generated_text + "\n")