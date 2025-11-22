# app/main.py
from app.model_loader import (
    load_plant_model,
    load_yolo_model,
    preprocess_image_from_bytes,
    predict_species,
    load_diagnosis_model,
    diagnose_state,
    IMG_SIZE
)


from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import traceback



class InferenceResponse(BaseModel):
    species: str
    confidence: float
    index: int

class PlantDiagnosisResponse(BaseModel):
    status: str
    confidence: float

app = FastAPI(title="Plant Inference API", description="YOLO + MobileNet 기반 식물 분류 및 진단 API")

MODEL_PATH = "saved_model/mobilenet_plant_classifier_final_savedmodel"
YOLO_PT_PATH = "saved_model/best1.pt"
PLANT_DIAG_MODEL_PATHS = {
    "관음죽": "saved_model/관음죽_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2"
}

plant_model = None
yolo_model = None
diagnosis_models = {}

# app/main.py (startup_event 함수)

@app.on_event("startup")
async def startup_event():
    global plant_model, yolo_model, diagnosis_models
    print("🚀 [System] 모델 로딩 시작…")
    try:
        # 1) plant_model 로드
        try:
            plant_model = load_plant_model(MODEL_PATH)
            print(f"[INFO] Plant 분류 모델 로드 성공: {MODEL_PATH}")
        except Exception as e:
            plant_model = None
            print(f"❌ [FATAL] Plant 분류 모델 로드 실패: {e}")
            print(traceback.format_exc())

        # 2) YOLO 모델 로드
        try:
            yolo_model = load_yolo_model(YOLO_PT_PATH)
            print(f"[INFO] YOLO 모델 로드 성공: {YOLO_PT_PATH}")
        except Exception as e:
            yolo_model = None
            print(f"❌ [FATAL] YOLO 모델 로드 실패: {e}")
            print(traceback.format_exc())

        # 3) 진단 모델 로드
        for species, path in PLANT_DIAG_MODEL_PATHS.items():
            try:
                print(f"[INFO] 진단 모델 로드 시도: species={species}, path={path}")
                diagnosis_models[species] = load_diagnosis_model(path)
                print(f"[INFO] '{species}' 진단 모델 로드 성공")
            except Exception as ex:
                diagnosis_models[species] = None
                # WARN 대신 FATAL로 표시하고 트레이스백 출력
                print(f"❌ [FATAL] '{species}' 진단 모델 로드 실패: {ex}")
                print(traceback.format_exc())

        print("✅ [System] 모든 모델 로딩 완료")

    except Exception as e:
        print(f"❌ [Error] 모델 로드 중 예외 발생: {e}")
        print(traceback.format_exc())
        plant_model = None
        yolo_model = None
        diagnosis_models = {}


@app.post("/classify", response_model=InferenceResponse)
async def classify_plant(image: UploadFile = File(...)):
    if plant_model is None or yolo_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="서버가 모델을 로드하지 못했습니다.")
    try:
        img_bytes = await image.read()
        print(f"[DEBUG] Uploaded filename: {image.filename!r}, size_bytes: {len(img_bytes)}")
        processed_image = preprocess_image_from_bytes(img_bytes, yolo_model=yolo_model, img_size=IMG_SIZE)
        try:
            import numpy as _np
            arr = processed_image
            if hasattr(arr, "shape"):
                print(f"[DEBUG] processed_image shape={arr.shape}, dtype={arr.dtype}, min={float(arr.min()):.6f}, max={float(arr.max()):.6f}")
        except Exception:
            pass
        result = predict_species(plant_model, processed_image)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"이미지 처리 실패: {e}")

@app.post("/diagnose", response_model=PlantDiagnosisResponse)
async def diagnose_plant(image: UploadFile = File(...)):
    if plant_model is None or yolo_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="서버가 모델을 로드하지 못했습니다.")
    try:
        img_bytes = await image.read()
        print(f"[DEBUG] diagnose uploaded file: {image.filename!r}, size={len(img_bytes)}")

        # species 판정 (기본 IMG_SIZE)
        processed_for_species = preprocess_image_from_bytes(img_bytes, yolo_model=yolo_model, img_size=IMG_SIZE)
        species_info = predict_species(plant_model, processed_for_species)
        inferred_species = species_info.get("species")
        species_conf = species_info.get("confidence")
        print(f"[INFO] 예측된 species: {inferred_species}, confidence={species_conf}")

        diag_model = diagnosis_models.get(inferred_species)
        if diag_model is None:
            print(f"[WARN] '{inferred_species}'에 대한 진단 모델이 존재하지 않음. UNKNOWN 반환.")
            return {"status": "UNKNOWN", "confidence": 0.0}

        # target_model 인자를 주면 전처리에서 expected shape를 사용
        # processed_for_diag 전처리
        processed_for_diag = preprocess_image_from_bytes(img_bytes, yolo_model=yolo_model,target_model=diag_model )

        try:
            import numpy as _np
            if hasattr(processed_for_diag, "shape"):
                print(f"[DEBUG] processed_for_diag shape={processed_for_diag.shape}")
        except Exception:
            pass

        diag_result = diagnose_state(diag_model, processed_for_diag)
        print(f"[INFO] diagnose result: {diag_result}")
        return diag_result

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"진단 처리 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
