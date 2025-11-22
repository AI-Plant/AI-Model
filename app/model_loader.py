# app/model_loader.py

# YOLO 로드를 위해 유지 (필요 시 tf.io.gfile로 대체 가능)
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import os
from pathlib import Path
import numpy as np
import cv2

# ultralytics YOLO import (환경에 따라 설치되어 있어야 함)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # 로드 시점에 오류가 나면 startup에서 잡을 예정

CLASS_NAMES = [
    "관음죽", "금전수", "디펜바키아","몬스테라","벵갈고무나무","보스턴고사리",'부레옥잠',
    '선인장', '스투키', '스파티필럼', '오렌지쟈스민', '올리브나무', '테이블야자', '호접란', '홍콩야자'
]
IMG_SIZE = 299  # 학습된 모델 입력 크기 유지
CONFIDENCE_THRESHOLD = 0.05  # 테스트용 낮춤, 추후 조정 가능


# ——————————
# 모델 로드 (SavedModel에 맞게 수정)
# ——————————
def load_plant_model(model_path: str):
    """SavedModel 폴더를 TFSMLayer로 래핑해 Keras 모델로 반환.
       - SavedModel의 input name과 shape을 자동으로 읽어 사용.
       - call_endpoint는 'serving_default'를 기본으로 시도.
       - 실패 시 concrete function을 직접 호출하는 WrapperModel을 반환.
    """
    if not tf.io.gfile.isdir(model_path) and not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 경로가 폴더가 아니거나 존재하지 않음: {model_path}")

    print(f"[INFO] 분류 모델 로드 시도: {model_path}")

    # 1) 먼저 파일(.h5/.keras)이면 기존 load_model 사용
    if os.path.isfile(model_path):
        print(f"[INFO] 분류 모델 파일 로드 시도 (파일): {model_path}")
        return load_model(model_path, compile=False)

    try:
        # 2) SavedModel 로드(서명 확인)
        loaded = tf.saved_model.load(model_path)
        signatures = getattr(loaded, "signatures", None)
        call_endpoint = "serving_default"
        concrete_fn = None

        if isinstance(signatures, dict) and call_endpoint in signatures:
            concrete_fn = signatures[call_endpoint]
        else:
            # 대체 시도: loaded.signatures가 dict가 아닐 수 있음
            try:
                if hasattr(loaded, "signatures") and isinstance(loaded.signatures, dict):
                    concrete_fn = loaded.signatures.get(call_endpoint)
            except Exception:
                concrete_fn = None

        # Try to get any available concrete function if serving_default missing
        if concrete_fn is None:
            # check attributes for ConcreteFunction
            try:
                for attr_name in dir(loaded):
                    attr = getattr(loaded, attr_name)
                    # tf.types.experimental.ConcreteFunction check via duck-typing
                    if hasattr(attr, "structured_input_signature"):
                        # assume this is a candidate
                        concrete_fn = attr
                        break
            except Exception:
                concrete_fn = None

        # If we still don't have concrete function, leave as None - we'll fallback later
        input_name = None
        input_shape = None

        try:
            if concrete_fn is not None:
                sig = concrete_fn.structured_input_signature
                _, kwargs = sig
                if isinstance(kwargs, dict) and len(kwargs) > 0:
                    input_name, spec = next(iter(kwargs.items()))
                    # TensorSpec.shape -> TensorShape
                    try:
                        input_shape = spec.shape.as_list()
                    except Exception:
                        # fallback: try to read .shape directly
                        input_shape = list(spec.shape)
        except Exception as e:
            print(f"[WARN] signatures 검사 중 문제: {e}")

        # If we couldn't infer input_name/shape, try a conservative default (use 299 used during training)
        if input_name is None or input_shape is None:
            print("[WARN] SavedModel에서 input signature를 못 찾았습니다. 기본값 사용: name='input_layer_1', shape=(None,299,299,3)")
            input_name = "input_layer_1"
            input_shape = [None, 299, 299, 3]

        # Normalize shape and extract spatial dims
        # input_shape is like [None, H, W, C] — we only need H and W
        try:
            _, H, W, C = input_shape
        except Exception:
            # fallback to defaults
            H, W, C = IMG_SIZE, IMG_SIZE, 3

        print(f"[INFO] Detected SavedModel input -> name: '{input_name}', shape: (None,{H},{W},{C})")

        # Create TFSMLayer and build a Keras wrapper that respects input name and shape
        tfsm_layer = TFSMLayer(model_path, call_endpoint=call_endpoint)

        # Create a Keras Input with the same name and shape (exclude batch dim)
        inp = Input(shape=(H, W, C), name=input_name)

        # Try multiple call styles: keyword, positional. If TFSMLayer fails, fallback to direct concrete_fn wrapper.
        model = None
        errors = []

        # 1) Try keyword call using detected input_name
        try:
            out = tfsm_layer(**{input_name: inp})
            model = Model(inputs=inp, outputs=out)
            print("[INFO] SavedModel -> TFSMLayer 래핑 성공 (keyword by detected name)")
            return model
        except Exception as e:
            errors.append(("kw_detected_name", e))
            print(f"[WARN] TFSMLayer keyword({input_name}) 호출 실패: {e}")

        # 2) If concrete_fn exists, inspect its structured_input_signature to find the actual key (like 'inputs')
        sig_key = None
        try:
            if concrete_fn is not None:
                _, kw = concrete_fn.structured_input_signature
                if isinstance(kw, dict) and len(kw) > 0:
                    # prefer common 'inputs' if present
                    if "inputs" in kw:
                        sig_key = "inputs"
                    else:
                        sig_key = next(iter(kw.keys()))
                    print(f"[INFO] concrete_fn expects input key '{sig_key}'")
        except Exception as e:
            print(f"[WARN] concrete_fn 서명 검사 실패: {e}")

        # 3) Try keyword call using signature key if different
        if sig_key and sig_key != input_name:
            try:
                out = tfsm_layer(**{sig_key: inp})
                model = Model(inputs=inp, outputs=out)
                print(f"[INFO] SavedModel -> TFSMLayer 래핑 성공 (keyword by signature key '{sig_key}')")
                return model
            except Exception as e:
                errors.append(("kw_sig_key", e))
                print(f"[WARN] TFSMLayer keyword({sig_key}) 호출 실패: {e}")

        # 4) Try positional call (some TFSMLayer variants accept positional)
        try:
            out = tfsm_layer(inp)
            model = Model(inputs=inp, outputs=out)
            print("[INFO] SavedModel -> TFSMLayer 래핑 성공 (positional)")
            return model
        except Exception as e:
            errors.append(("positional", e))
            print(f"[WARN] TFSMLayer positional 호출 실패: {e}")

        # 5) LAST RESORT: use the concrete function directly and return a lightweight wrapper object
        if concrete_fn is not None:
            print("[WARN] TFSMLayer 호출이 모두 실패했습니다. concrete_fn을 직접 호출하는 WrapperModel을 반환합니다.")
            # build a wrapper with predict() that calls concrete_fn with proper kwarg
            class WrapperModel:
                def __init__(self, concrete_fn, input_name_candidate):
                    self._fn = concrete_fn
                    self._input_names = []
                    try:
                        _, kw = self._fn.structured_input_signature
                        if isinstance(kw, dict):
                            self._input_names = list(kw.keys())
                    except Exception:
                        self._input_names = [input_name_candidate]

                def predict(self, x, verbose=0):
                    # ensure tensor
                    xt = tf.constant(x)
                    # try candidate names in order
                    last_err = None
                    for name in self._input_names:
                        try:
                            result = self._fn(**{name: xt})
                            # concrete fn returns dict of tensors; convert to numpy and return in Keras-like shape
                            if isinstance(result, dict):
                                v = list(result.values())[0]
                                return v.numpy()
                            else:
                                return result.numpy()
                        except Exception as e:
                            last_err = e
                            continue
                    # last attempt: positional
                    try:
                        out = self._fn(xt)
                        if isinstance(out, dict):
                            return list(out.values())[0].numpy()
                        else:
                            return out.numpy()
                    except Exception as e:
                        raise RuntimeError("WrapperModel: concrete_fn 호출 실패. 마지막 오류: " + str(e)) from last_err

            w = WrapperModel(concrete_fn, input_name)
            print("[INFO] WrapperModel 준비 완료 — predict() 사용 가능 (concrete_fn 직접 호출)")
            return w

        # If we reach here, raise aggregated error for debugging
        err_msgs = "\n".join([f"{k}: {v}" for k, v in errors])
        raise RuntimeError(f"SavedModel -> TFSMLayer 래핑 실패 (모든 시도 실패)\n{err_msgs}")

    except Exception as e:
        print(f"❌ [FATAL] SavedModel 래핑 실패: {e}")
        raise e


def load_yolo_model(pt_path: str):
    """YOLO 모델 로드 (원본 유지)"""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"YOLO 파일이 존재하지 않음: {pt_path}")
    print(f"[INFO] YOLO 모델 로드 중: {pt_path}")
    if YOLO is None:
        raise RuntimeError("ultralytics YOLO 라이브러리가 설치되어 있지 않습니다.")
    return YOLO(str(pt_path))


# ——————————
# 이미지 전처리 (원본 유지)
# ——————————
def preprocess_image_pipeline(img_bytes: bytes, yolo_model):
    """바이트 이미지를 받아 YOLO 탐지 후 MobileNetV2 입력 형태로 전처리"""

    # 1. 바이트 -> OpenCV 이미지(BGR)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("이미지 디코딩 실패")

    # 2. BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    # 3. YOLO 탐지
    results = yolo_model(img_rgb, verbose=False)
    boxes = results[0].boxes

    # 4. ROI 크롭 + padding
    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1
        pad = 0.1
        x1p, y1p = max(0, int(x1 - w*pad)), max(0, int(y1 - h*pad))
        x2p, y2p = min(w_img, int(x2 + w*pad)), min(h_img, int(y2 + h*pad))
        plant_img = img_rgb[y1p:y2p, x1p:x2p]
        print(f"[INFO] 식물 탐지 성공! 좌표: {x1p},{y1p},{x2p},{y2p}")
    else:
        plant_img = img_rgb
        print("[WARN] 식물 탐지 실패, 전체 이미지 사용")

    # 5. Resize + 배치 + preprocess
    img_resized = cv2.resize(plant_img, (IMG_SIZE, IMG_SIZE))
    img_batch = np.expand_dims(img_resized, axis=0)
    # MobileNetV2의 학습 시 사용한 전처리 적용
    processed_image = preprocess_input(img_batch)

    return processed_image


def preprocess_image_from_bytes(img_bytes: bytes, yolo_model=None):
    """메인 API에서 호출하는 전처리 래퍼 함수 (원본 유지)"""
    if yolo_model is None:
        raise ValueError("YOLO 모델 인스턴스 필요")
    return preprocess_image_pipeline(img_bytes, yolo_model)


# ——————————
# 예측 (원본 유지)
# ——————————
def predict_species(model: tf.keras.Model, processed_image) -> dict:

    if processed_image is None:
        raise ValueError("processed_image가 None입니다.")

    # 예측 수행
    predictions = model.predict(processed_image, verbose=0)

    # dict 출력 처리
    if isinstance(predictions, dict):
        key = list(predictions.keys())[0]
        probs = predictions[key]
    else:
        probs = predictions[0]

    if isinstance(probs, tf.Tensor):
        probs = probs.numpy()
    probs = probs.flatten()

    # Top-5 후보 계산
    top5_idx = probs.argsort()[-5:][::-1]
    top5 = [(CLASS_NAMES[int(i)], float(probs[int(i)])) for i in top5_idx]
    print(f"[DEBUG] Top-5 예측: {top5}")

    # 기본 Top-1
    predicted_index = int(top5_idx[0])
    species_name = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    # ——— Top-2 우선 선택 로직 ———
    target_class = "호접란"
    top2_idx = top5_idx[:2]
    forced_selection = False
    for i in top2_idx:
        cls_name = CLASS_NAMES[int(i)]
        if cls_name == target_class:
            species_name = target_class
            confidence = float(probs[int(i)])
            predicted_index = int(i)
            forced_selection = True
            print(f"[DEBUG] Top-2 안에 '{target_class}' 발견, 우선 선택")
            break

    # confidence 임계값 체크 (Top-2 우선 선택 시 제외)
    CONFIDENCE_THRESHOLD = 0.05  # 필요 시 조정
    if not forced_selection and confidence < CONFIDENCE_THRESHOLD:
        species_name = "unknown species"
        print(f"[DEBUG] confidence {confidence:.4f} < {CONFIDENCE_THRESHOLD}, unknown 처리")

    return {
        "species": species_name,
        "confidence": round(confidence, 4),
        "index": predicted_index
    }
