# 필요한 라이브러리들을 불러옵니다
import os  # 운영 체제와 관련된 기능을 사용하기 위한 라이브러리
import tkinter as tk  # GUI(그래픽 사용자 인터페이스)를 만들기 위한 라이브러리
from tkinter import filedialog  # 파일 선택 대화상자를 사용하기 위한 모듈
import PyPDF2  # PDF 파일을 다루기 위한 라이브러리
import re  # 정규 표현식을 사용하기 위한 라이브러리
import json  # JSON 형식의 데이터를 다루기 위한 라이브러리
import fitz  # PyMuPDF 라이브러리, PDF 파일의 메타데이터와 목차를 추출하기 위해 사용

# PDF 파일을 텍스트로 변환하고 관련 정보를 추출하는 함수
def convert_pdf_to_text():
    # 사용자가 PDF 파일을 선택할 수 있는 대화상자를 엽니다
    file_path = filedialog.askopenfilename(filetypes=[("PDF 파일", "*.pdf")])
    if file_path:  # 파일이 선택되었다면
        # PDF 파일을 열고 내용을 읽습니다
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)  # PDF의 총 페이지 수를 구합니다
            text = ''
            # 모든 페이지의 텍스트를 추출합니다
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # 추출된 텍스트의 공백을 정리합니다
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 텍스트를 문장 단위로 나눕니다
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            # 텍스트를 1000자 이하의 청크로 나눕니다
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 1000:
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            
            # 청크를 vault.txt 파일에 저장합니다
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    vault_file.write(chunk.strip() + "\n")
            
            # PDF 메타데이터를 추출합니다
            doc = fitz.open(file_path)
            metadata = doc.metadata
            # 메타데이터를 JSON 파일로 저장합니다
            with open("pdf_metadata.json", "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
            
            # PDF 목차를 추출합니다
            toc = doc.get_toc()
            # 목차를 텍스트 파일로 저장합니다
            with open("pdf_toc.txt", "w", encoding="utf-8") as toc_file:
                for item in toc:
                    toc_file.write(f"{'  ' * (item[0] - 1)}{item[1]} (페이지: {item[2]})\n")
            
            # 작업 완료 메시지를 출력합니다
            print(f"PDF 내용이 vault.txt에 추가되었습니다. 각 청크는 별도의 줄에 저장되었습니다.")
            print(f"PDF 메타데이터가 pdf_metadata.json에 저장되었습니다.")
            print(f"PDF 목차가 pdf_toc.txt에 저장되었습니다.")

# 텍스트 파일을 업로드하고 처리하는 함수
def upload_txtfile():
    # 사용자가 텍스트 파일을 선택할 수 있는 대화상자를 엽니다
    file_path = filedialog.askopenfilename(filetypes=[("텍스트 파일", "*.txt")])
    if file_path:  # 파일이 선택되었다면
        # 텍스트 파일을 열고 내용을 읽습니다
        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text = txt_file.read()
            
            # 텍스트의 공백을 정리합니다
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 텍스트를 문장 단위로 나누고 1000자 이하의 청크로 나눕니다
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 1000:
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            
            # 청크를 vault.txt 파일에 저장합니다
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    vault_file.write(chunk.strip() + "\n")
            print(f"텍스트 파일 내용이 vault.txt에 추가되었습니다. 각 청크는 별도의 줄에 저장되었습니다.")
# JSON 파일을 업로드하고 처리하는 함수
def upload_jsonfile():
    # 사용자가 JSON 파일을 선택할 수 있는 대화상자를 엽니다
    file_path = filedialog.askopenfilename(filetypes=[("JSON 파일", "*.json")])
    if file_path:  # 파일이 선택되었다면
        # JSON 파일을 열고 내용을 읽습니다
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
            
            # JSON 데이터를 문자열로 변환합니다
            text = json.dumps(data, ensure_ascii=False)
            
            # 텍스트의 공백을 정리합니다
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 텍스트를 문장 단위로 나누고 1000자 이하의 청크로 나눕니다
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 1000:
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            
            # 청크를 vault.txt 파일에 저장합니다
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    vault_file.write(chunk.strip() + "\n")
            print(f"JSON 파일 내용이 vault.txt에 추가되었습니다. 각 청크는 별도의 줄에 저장되었습니다.")

# GUI 창을 생성합니다
root = tk.Tk()
root.title("파일 업로드 (.pdf, .txt, or .json)")

# PDF 업로드 버튼을 생성합니다
pdf_button = tk.Button(root, text="PDF 업로드", command=convert_pdf_to_text)
pdf_button.pack(pady=10)

# 텍스트 파일 업로드 버튼을 생성합니다
txt_button = tk.Button(root, text="텍스트 파일 업로드", command=upload_txtfile)
txt_button.pack(pady=10)

# JSON 파일 업로드 버튼을 생성합니다
json_button = tk.Button(root, text="JSON 파일 업로드", command=upload_jsonfile)
json_button.pack(pady=10)

# GUI 이벤트 루프를 시작합니다
root.mainloop()