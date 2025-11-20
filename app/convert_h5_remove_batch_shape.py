# convert_h5_remove_batch_shape.py
import traceback
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import h5py

# 설정: 원본 .h5 경로와 출력 경로
SRC_H5 = Path("../saved_model/mobilenet_plant_classifier_final.h5")
OUT_H5 = Path("../saved_model/mobilenet_plant_classifier_final_nobatch.h5")

IMG_SIZE = 299
NUM_CLASSES = 15  # 네 CLASS_NAMES 길이에 맞춰 바꿔. 예: 15
# 만약 CLASS_NAMES가 다르면 이 값을 정확히 맞춰줘.

def create_model(num_classes=NUM_CLASSES, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights=None,
        pooling='avg'
    )
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    return model

def try_load_full_model(src_path):
    """시도 1: load_model(..., compile=False)"""
    try:
        print("[Step 1] load_model(..., compile=False) 시도 중...")
        model = keras.models.load_model(str(src_path), compile=False)
        print("[Step 1] load_model 성공!")
        return model
    except Exception as e:
        print("[Step 1] load_model 실패:", e)
        return None

def wrap_model_with_new_input(old_model):
    """old_model을 새로운 Input(shape=(299,299,3))로 감싸서 새 모델 생성"""
    try:
        new_input = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        # old_model(old_input) 형태로 호출 가능하면 그 결과를 new model의 output으로 사용
        new_output = old_model(new_input)
        new_model = keras.Model(inputs=new_input, outputs=new_output)
        print("[Step 1b] 새 Input으로 래핑 성공.")
        return new_model
    except Exception as e:
        print("[Step 1b] 래핑 실패:", e)
        return None

def try_create_and_load_weights(src_path):
    """시도 2: create_model()로 구조 생성 후 load_weights(by_name=True) 시도"""
    try:
        print("[Step 2] create_model() 생성 및 load_weights(by_name=True) 시도 중...")
        model = create_model()
        # load_weights는 파일 저장 방식에 따라 by_name으로 로드해볼 수 있음
        model.load_weights(str(src_path), by_name=True)
        print("[Step 2] weights 로드(이름 기준) 성공 가능성 있음 - 검증 필요")
        return model
    except Exception as e:
        print("[Step 2] weights-only 로드 실패:", e)
        return None

def inspect_h5(src_path):
    print("[Inspect] h5 내부 구조 확인 중...")
    try:
        with h5py.File(str(src_path), 'r') as f:
            print("Top keys:", list(f.keys()))
            print("Attrs:", list(f.attrs.keys()))
            print("'model_config' in file?", 'model_config' in f)
            if 'model_config' in f:
                try:
                    raw = f['model_config'][()]
                    print("model_config type:", type(raw))
                except Exception as ex:
                    print("model_config read error:", ex)
            if 'layer_names' in f:
                print("layer_names (count):", len(f['layer_names']))
    except Exception as e:
        print("h5 inspect 실패:", e)

def save_model_safely(model, out_path):
    # overwrite if exists
    if out_path.exists():
        out_path.unlink()
    model.save(str(out_path))
    print(f"[Saved] 새 파일 저장: {out_path}")

def main():
    if not SRC_H5.exists():
        print("원본 h5 파일을 찾을 수 없음:", SRC_H5)
        return

    # 1) load_model 시도
    full = try_load_full_model(SRC_H5)
    if full is not None:
        # 만약 로드가 된 모델의 InputLayer가 batch_shape로 인해 로드된 상태라면
        # 우리는 새 Input으로 래핑해서 batch_shape 없는 모델로 저장
        wrapped = wrap_model_with_new_input(full)
        if wrapped is not None:
            save_model_safely(wrapped, OUT_H5)
            print("[Done] 변환 완료 (방법: load_model -> wrap)")
            return
        else:
            # 래핑 실패하면 다음 단계로
            print("[Warn] 래핑 실패. 다음 방법 시도...")
    
    # 2) create_model() + load_weights(by_name=True)
    weights_model = try_create_and_load_weights(SRC_H5)
    if weights_model is not None:
        # 간단한 검증: 레이어 수와 마지막 레이어 이름 확인
        print("Model summary (short):")
        weights_model.summary()
        save_model_safely(weights_model, OUT_H5)
        print("[Done] 변환 완료 (방법: create_model + load_weights(by_name=True))")
        return

    # 3) h5 내부 정보 보여주고 수동 조치 가이드
    inspect_h5(SRC_H5)
    print("""
[Fail] 자동 변환 실패.
다음 조치 안내:
- 1) 학습 환경에서 아래 코드로 새 모델을 재저장하세요(권장):
    inputs = keras.Input(shape=(299,299,3))
    outputs = old_model(inputs)   # old_model: 로드 가능한 모델
    new_model = keras.Model(inputs, outputs)
    new_model.save("mobilenet_clean.h5")

- 2) 또는 학습 스크립트에서 Input을 shape=(299,299,3)으로 바꿔서 다시 저장하세요.

원하면 내가 직접 너의 h5를 분석해 더 깊게 도와줄게. 위 출력 결과(특히 h5 keys)를 여기에 붙여줘.
""")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
