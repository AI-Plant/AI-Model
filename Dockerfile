# 1. 베이스 이미지: TensorFlow와 Python이 포함된 이미지 사용
FROM python:3.10-slim

# **** 이 부분에 시스템 라이브러리 설치를 추가합니다 ****
# OpenCV (cv2) 실행에 필요한 기본 GL/X11 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*
# ******************************************************

# 2. 환경 변수 설정
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# 3. Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 4. 모델 파일 복사 (모델 파일이 크므로 별도로 관리)
# 모델 파일이 크다면, Dockerfile COPY 대신 S3/GCP 등 외부 스토리지에서 다운로드하도록 변경하는 것이 좋습니다.
COPY saved_model/ $APP_HOME/saved_model/

# 5. 애플리케이션 코드 복사
COPY app/ $APP_HOME/app/

# 6. 서버 실행 명령어: Uvicorn을 사용하여 Gunicorn의 워커 프로세스를 관리하는 방식(권장)
# Spring의 WebClient가 연결할 포트 8000 노출
EXPOSE 8000

# Gunicorn + Uvicorn 설정 (프로덕션 배포에 권장되는 안정적인 방식)
CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]

# 'gunicorn'이 번거롭다면 단순하게 CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 사용 가능
