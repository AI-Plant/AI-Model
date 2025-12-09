from app.model_loader import (
    load_plant_model,
    load_yolo_model,
    preprocess_image_from_bytes,
    predict_species,
    load_yolo_model_for_species,
    load_diagnosis_model,
    diagnose_state,
    IMG_SIZE
)
from app.diagnose_service import detect_leaf_and_crop_by_species, preprocess_whole_image 
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from app.middleware import StripPathMiddleware
from pydantic import BaseModel
import traceback
import numpy as np
import cv2


class InferenceResponse(BaseModel):
    species: str
    confidence: float
    index: int

class PlantDiagnosisResponse(BaseModel):
    status: str
    confidence: float

app = FastAPI(title="Plant Inference API", description="YOLO + MobileNet 기반 식물 분류 및 진단 API")
app.add_middleware(StripPathMiddleware)

MODEL_PATH = "saved_model/mobilenet_plant_classifier_final_v2"
YOLO_PT_PATH = "saved_model/best1.pt"
PLANT_DIAG_MODEL_PATHS = {
    "관음죽": "saved_model/mobilenetv3_log/관음죽_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "금전수": "saved_model/mobilenetv3_log/금전수_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "디펜바키아": "saved_model/mobilenetv3_log/디펜바키아_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "몬스테라": "saved_model/mobilenetv3_log/몬스테라_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "벵갈고무나무": "saved_model/mobilenetv3_log/벵갈고무나무_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "보스턴고사리": "saved_model/mobilenetv3_log/보스턴고사리_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "부레옥잠": "saved_model/mobilenetv3_log/부레옥잠_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "선인장" : "saved_model/mobilenetv3_log/선인장_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "스투키" : "saved_model/mobilenetv3_log/스투키_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "스파티필럼" : "saved_model/mobilenetv3_log/스파티필럼_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "오렌지쟈스민" : "saved_model/mobilenetv3_log/오렌지쟈스민_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "올리브나무" : "saved_model/mobilenetv3_log/올리브나무_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "테이블야자" : "saved_model/mobilenetv3_log/테이블야자_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "호접란" : "saved_model/mobilenetv3_log/호접란_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2",
    "홍콩야자":  "saved_model/mobilenetv3_log/홍콩야자_mobilenetv3_large_best(final)_savedmodel_FINAL_K3_V2"
}
YOLO_PATHS_STAGE1 = {
    "관음죽": "saved_model/yolo_log/관음죽_yolo-20251122T123532Z-1-001/관음죽_yolo/detect/stage2_1280_ft/weights/best.pt",
    "금전수": "saved_model/yolo_log/금전수_yolo/detect/stage2_1280_ft/weights/best.pt",
    "스파티필럼": "saved_model/yolo_log/스파티필럼_yolo/detect/stage2_1280_ft/weights/best.pt",
    "디펜바키아": "saved_model/yolo_log/디펜바키아_yolo/detect/stage2_1280_ft/weights/best.pt",
    "몬스테라": "saved_model/yolo_log/몬스테라_yolo/detect/stage2_1280_ft/weights/best.pt",
    "벵갈고무나무": "saved_model/yolo_log/벵갈고무나무_yolo/detect/stage2_1280_ft/weights/best.pt",
    "보스턴고사리": "saved_model/yolo_log/보스턴고사리_yolo/detect/stage2_1280_ft/weights/best.pt",
    "부레옥잠": "saved_model/yolo_log/부레옥잠_yolo/detect/stage2_1280_ft/weights/best.pt",
    "선인장": "saved_model/yolo_log/선인장_yolo/detect/stage2_1280_ft/weights/best.pt",
    "스투키": "saved_model/yolo_log/스투키_yolo/detect/stage2_1280_ft/weights/best.pt",
    "오렌지쟈스민": "saved_model/yolo_log/오렌지쟈스민_yolo/detect/stage2_1280_ft/weights/best.pt",
    "올리브나무": "saved_model/yolo_log/올리브나무_yolo/detect/stage2_1280_ft/weights/best.pt",
    "테이블야자": "saved_model/yolo_log/테이블야자_yolo/detect/stage2_1280_ft/weights/best.pt",
    "호접란": "saved_model/yolo_log/호접란_yolo/detect/stage2_1280_ft/weights/best.pt",
    "홍콩야자": "saved_model/yolo_log/홍콩야자_yolo/detect/stage2_1280_ft/weights/best.pt"
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


@app.post("/diagnosee", response_model=PlantDiagnosisResponse)
async def diagnose_plant(request: Request, image: UploadFile = File(...)):
    url_path = request.url.path.strip()
    if url_path != "/diagnose":
        print(f"[WARN] URL 끝 공백/개행 감지: {request.url.path}")
    else:
        print(f"[INFO] URL 정상: {url_path}")

    if plant_model is None or yolo_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="모델 미준비")

    # 1) 이미지 읽기
    img_bytes = await image.read()

    # 2) 종 분류
    processed_image = preprocess_image_from_bytes(img_bytes, yolo_model=yolo_model, img_size=IMG_SIZE)
    species_info = predict_species(plant_model, processed_image)
    inferred_species = species_info.get("species")
    print(f"[INFO] classify -> inferred_species={inferred_species}")

    # 3) 진단 모델 확인
    diag_model = diagnosis_models.get(inferred_species)
    if diag_model is None:
        print(f"[WARN] {inferred_species} 진단 모델 없음")
        return {"status": "UNKNOWN", "confidence": 0.0}

    # 4) species별 YOLO 모델로 crop 시도 (예외 처리 + 폴백)
    yolo_for_crop = load_yolo_model_for_species(inferred_species) or yolo_model

    # 이미지 decode
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    # crop 수행 (예외 처리 + 폴백)
    crop_rgb = None
    whole_proc = None
    try:
        # 테스트 시에는 conf_threshold들을 내부에서 재시도하도록 detect 함수가 처리함
        # fallback_yolo_model으로 global yolo_model을 전달하여 species YOLO 실패 시 재시도 하게 함
        crop_rgb = detect_leaf_and_crop_by_species(img_bgr, inferred_species, yolo_model=yolo_for_crop, fallback_yolo_model=yolo_model)
    except HTTPException as he:
        print(f"[WARN] detect_leaf_and_crop_by_species HTTPException: {he.detail}")
        # 폴백: 전체 이미지로 전처리 시도 (diagnosis input 크기 맞추기 전 처리)
        try:
            print("[INFO] 잎 검출 실패 -> 전체 이미지를 사용하여 진단 시도")
            whole_proc = preprocess_whole_image(img_bytes, img_size=IMG_SIZE)
            crop_rgb = None  # signal: use whole_proc path
        except Exception as e:
            print(f"[ERROR] 전체 이미지 전처리 실패: {e}")
            raise HTTPException(status_code=400, detail=f"잎 검출 및 전체 이미지 전처리 실패: {e}")
    except Exception as e:
        import traceback as _tb
        print("[ERROR] detect_leaf_and_crop_by_species 예외 발생:")
        print(_tb.format_exc())
        raise HTTPException(status_code=400, detail=f"잎 탐지 중 예외: {e}")

    # 5) resize + preprocess (diag_model의 기대 shape 사용)
    exp = getattr(diag_model, "_expected_input_shape", (224, 224, 3))

    try:
        if crop_rgb is not None:
            crop_resized = cv2.resize(crop_rgb, (exp[1], exp[0]))
            crop_batch = np.expand_dims(crop_resized, axis=0)
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _pp
            crop_proc = _pp(crop_batch.astype("float32"))
        else:
            # whole_proc 경로 사용
            if whole_proc is None:
                whole_proc = preprocess_whole_image(img_bytes, img_size=IMG_SIZE)
            # whole_proc 은 (1,H,W,C) 형태. 필요하면 resize to expected shape
            if whole_proc.shape[1] != exp[0] or whole_proc.shape[2] != exp[1]:
                import numpy as _np
                batch_imgs = []
                for i in range(whole_proc.shape[0]):
                    img = whole_proc[i]
                    mn, mx = img.min(), img.max()
                    if mx - mn <= 0:
                        img_uint8 = (img * 255).astype("uint8")
                    else:
                        img_uint8 = (((img - mn) / (mx - mn)) * 255).astype("uint8")
                    img_resz = cv2.resize(img_uint8, (exp[1], exp[0]))
                    batch_imgs.append(img_resz)
                crop_batch = _np.stack(batch_imgs, axis=0).astype("float32")
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _pp
                crop_proc = _pp(crop_batch)
            else:
                crop_proc = whole_proc
    except Exception as e:
        print(f"[ERROR] 진단 입력 준비 실패: {e}")
        import traceback as _tb
        print(_tb.format_exc())
        raise HTTPException(status_code=400, detail=f"진단 입력 준비 실패: {e}")

    # 6) 진단 수행
    try:
        diag_result = diagnose_state(diag_model, crop_proc)
        print(f"[INFO] diagnose result: {diag_result}")
        return diag_result
    except Exception as e:
        print(f"[ERROR] diagnose_state 실패: {e}")
        import traceback as _tb
        print(_tb.format_exc())
        raise HTTPException(status_code=400, detail=f"진단 수행 중 오류: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
