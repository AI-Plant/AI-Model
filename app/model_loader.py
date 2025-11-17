import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras import Model
import numpy as np
import json
import base64
from io import BytesIO
from pathlib import Path

# --- 설정값 ---
CLASS_NAMES = [
    "관음죽", "금전수", "디펜바키아", "드라세나", "몬스테라",
    "벵갈고무나무", "산세베리아", "스킨답서스", "수채화고무나무", "셀렘",
    "아레카야자", "아이비", "여인초", "테이블야자", "필로덴드론"
]
IMG_SIZE = 299  # 모델 입력 크기
CONFIDENCE_THRESHOLD = 0.85

# --- DTypePolicy 에러 대응 ---
DTypePolicy = Policy

# ---------------------
# 모델 구조 생성 (MobileNetV2 기반)
# ---------------------
def create_model(num_classes=len(CLASS_NAMES), input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights=None,           # 나중에 load_weights 사용
        pooling='avg'           # GlobalAveragePooling2D 적용
    )
    outputs = Dense(num_classes, activation='softmax', name='predictions')(base.output)
    model = Model(inputs=base.input, outputs=outputs)
    return model

# ---------------------
# h5 파일 안전 로드
# ---------------------
def load_plant_model_from_h5(h5_path: str):
    from tensorflow.keras.models import load_model
    from tensorflow.keras import Model
    import tensorflow as tf

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"h5 파일이 존재하지 않음: {h5_path}")

    # 전체 모델 로드는 건너뛰고 weights-only로 바로 처리
    model = create_model()  # create_model()에서 MobileNetV2 구조 정의
    try:
        model.load_weights(h5_path, by_name=True)  # 레이어 이름 기준으로 로드
        print(f"[INFO] weights-only 로드 성공: {h5_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"weights-only 로드 실패: {e}")


# ---------------------
# JSON 관련 유틸
# ---------------------
def _remove_keys_recursively(obj, keys_to_remove):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in keys_to_remove:
                del obj[k]
            else:
                _remove_keys_recursively(obj[k], keys_to_remove)
    elif isinstance(obj, list):
        for item in obj:
            _remove_keys_recursively(item, keys_to_remove)

def find_keys_in_json(obj, key_name, path=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            cur_path = f"{path}/{k}"
            if k == key_name:
                found.append((cur_path, v))
            found.extend(find_keys_in_json(v, key_name, cur_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            found.extend(find_keys_in_json(item, key_name, f"{path}[{idx}]"))
    return found

def clean_model_json_str(model_json_str: str, keys_to_remove=None) -> str:
    if keys_to_remove is None:
        keys_to_remove = ['synchronized']
    parsed = json.loads(model_json_str)
    _remove_keys_recursively(parsed, keys_to_remove)
    return json.dumps(parsed)

# ---------------------
# JSON 구조 + 가중치 로드
# ---------------------
def load_plant_model(config_path: str, weights_path: str):
    custom_objects = {'DTypePolicy': DTypePolicy}
    # 1. JSON 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_json = f.read()

    # 2. 문제 키 제거
    cleaned_json = clean_model_json_str(raw_json, keys_to_remove=['synchronized'])

    # 3. 모델 구조 생성
    try:
        model = model_from_json(cleaned_json, custom_objects=custom_objects)
    except Exception as e:
        raise RuntimeError(f"model_from_json 실패: {e}")

    # 4. 가중치 로드
    try:
        model.load_weights(weights_path)
    except Exception as e:
        raise RuntimeError(f"가중치 로드 실패: {e}")

    print("[INFO] 모델 로드 완료")
    return model

# ---------------------
# 이미지 전처리
# ---------------------
def preprocess_image_from_base64(image_base64_string: str):
    if not image_base64_string:
        raise ValueError("이미지 문자열이 비어있습니다.")
    img_data = base64.b64decode(image_base64_string)
    img = image.load_img(BytesIO(img_data), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype('float32'))
    return img_array

# ---------------------
# 예측
# ---------------------
def predict_species(model: tf.keras.Model, processed_image) -> dict:
    if processed_image is None:
        raise ValueError("processed_image가 None입니다.")
    predictions = model.predict(processed_image)
    probs = tf.nn.softmax(predictions[0]).numpy()
    confidence = float(np.max(probs))
    predicted_index = int(np.argmax(probs))
    species_name = CLASS_NAMES[predicted_index] if confidence >= CONFIDENCE_THRESHOLD else "unknown species"
    return {
        "species": species_name,
        "confidence": round(confidence, 4),
        "index": predicted_index
    }
def preprocess_image_from_bytes(img_bytes: bytes):
    img = image.load_img(BytesIO(img_bytes), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype('float32'))
    return img_array
