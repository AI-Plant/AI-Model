from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional
# h5 자동 로드 함수 가져오기
from .model_loader import load_plant_model_from_h5, preprocess_image_from_base64, predict_species, preprocess_image_from_bytes

# 1. Pydantic 모델 정의
class InferenceRequest(BaseModel):
    image_base64: str

class InferenceResponse(BaseModel):
    species: str
    confidence: float

# 2. FastAPI 인스턴스 및 모델 로드
app = FastAPI(
    title="Plant Inference API",
    description="식물 이미지 분류 모델 (MobileNetV2) 추론 API"
)

# h5 파일 경로: 실제 파일 이름/경로로 바꿔주세요.
# 예: saved_model/mobilenet_plant_classifier.h5 또는 saved_model/model.weights.h5
MODEL_H5_PATH = "saved_model/mobilenet_plant_classifier.h5"
plant_model = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델을 로드합니다."""
    global plant_model
    try:
        plant_model = load_plant_model_from_h5(MODEL_H5_PATH)
    except Exception as e:
        # 운영 환경에서는 예외를 던져 프로세스를 종료시키는 것이 안전하지만
        # 개발 중에는 로그를 남기고 계속 실행하도록 할 수 있습니다.
        print(f"모델 로드 중 오류 발생: {e}")
        plant_model = None

# 3. 엔드포인트 정의 (POST /classify)
@app.post(
    "/classify",
    response_model=InferenceResponse,
    summary="식물 이미지 분류 추론",
    description="이미지 파일(jpg/png/webp)을 받아 분류합니다."
)
async def classify_plant(image: UploadFile = File(...)):
    if plant_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 아직 로드되지 않았습니다."
        )

    try:
        # 1) 이미지 raw bytes 읽기
        img_bytes = await image.read()

        # 2) 전처리
        processed_image = preprocess_image_from_bytes(img_bytes)

        # 3) 추론 수행
        result = predict_species(plant_model, processed_image)

        return InferenceResponse(**result)

    except Exception as e:
        print(f"추론 요청 처리 중 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"이미지 처리 실패: {e}"
        )
