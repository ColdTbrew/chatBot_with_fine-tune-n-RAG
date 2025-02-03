# python 3.10 기반
FROM python:3.10
# .bashrc 파일에 별칭 추가 (ll 명령 입력 시 ls -l 실행)
RUN echo "alias ll='ls -l'" >> ~/.bashrc
# container의 usr/src 경로에서 진행
WORKDIR /usr/src
# requirements.txt를 복사해서 넣기
COPY ./requirements.txt /usr/src/requirements.txt
# app 디렉토리를 복사해서 넣기
COPY ./app /usr/src/app
# pip install requirements.txt 실행
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# FastAPI uvicorn으로 실행 (도커 실행시)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]