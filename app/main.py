from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
# ⭐️ 1. 로드 함수 이름 변경: load_plant_model_from_h5 -> load_plant_model
from .model_loader import load_plant_model, load_yolo_model, predict_species, preprocess_image_from_bytes
import traceback
import sys

# ——————————
## ⚙️ FastAPI 설정
# ——————————
class InferenceResponse(BaseModel):
    species: str
    confidence: float
    index: int

app = FastAPI(title="Plant Inference API", description="YOLO + MobileNetV2 기반 식물 분류 API")

# ⭐️ 2. 모델 경로를 SavedModel 폴더 경로로 변경
# 서버 개발자가 다운로드한 SavedModel 폴더의 상대 경로를 지정합니다.
MODEL_PATH = "saved_model/mobilenet_plant_classifier_final_savedmodel" 
YOLO_PT_PATH = "saved_model/best1.pt"

plant_model = None
yolo_model = None

@app.on_event("startup")
async def startup_event():
    global plant_model, yolo_model
    print("🚀 [System] 모델 로딩 시작…")
    try:
        # ⭐️ 3. 변경된 함수 이름 및 SavedModel 경로를 사용하여 모델 로드
        plant_model = load_plant_model(MODEL_PATH)
        yolo_model = load_yolo_model(YOLO_PT_PATH)
        print("✅ [System] 모델 로딩 완료!")
    except Exception as e:
        print(f"❌ [Error] 모델 로드 실패: {e}")
        plant_model = None
        yolo_model = None

# ——————————
## 🌐 엔드포인트
# ——————————
@app.post("/classify", response_model=InferenceResponse)
async def classify_plant(image: UploadFile = File(...)):
    if plant_model is None or yolo_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="서버가 모델을 로드하지 못했습니다.")

    try:
        img_bytes = await image.read()

        # --- 추가 디버깅 로그: 업로드된 파일 정보 ---
        print(f"[DEBUG] Uploaded filename: {image.filename!r}, content_type: {image.content_type!r}, size_bytes: {len(img_bytes)}")

        # --- 전처리 직전/직후 로그 ---
        processed_image = preprocess_image_from_bytes(img_bytes, yolo_model=yolo_model)
        try:
            import numpy as _np
            print(f"[DEBUG] processed_image type={type(processed_image)}, shape={getattr(processed_image,'shape',None)}, dtype={getattr(processed_image,'dtype',None)}")
            # sample min/max
            arr = processed_image if isinstance(processed_image, (list, tuple)) else processed_image
            if hasattr(arr, "shape"):
                print(f"[DEBUG] processed_image sample min/max: {float(arr.min()):.6f}/{float(arr.max()):.6f}")
        except Exception as _err:
            print("[WARN] processed_image 디버깅 정보 얻기 실패:", _err)

        # --- 예측 시도 및 상세 예외 캡처 ---
        result = predict_species(plant_model, processed_image)
        return result

    except Exception as e:
        # 상세한 트레이스백을 찍는다 (핵심)
        tb = traceback.format_exc()
        print("⚠️ [ERROR] classify_plant 예외 발생:", repr(e))
        print("⚠️ [TRACEBACK]\n", tb)
        # 클라이언트에선 간단히 400을 주되, 내부 로그에는 전체 트레이스 출력됨
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"이미지 처리 실패: {e}")