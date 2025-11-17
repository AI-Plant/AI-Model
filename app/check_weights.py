
import h5py
from tensorflow.keras.applications import MobileNet  
h5_path = r"C:\smp\SmPlant\AI-Model\saved_model\mobilenet_plant_classifier.h5"

# 1) saved file 내부 구조 확인 + model_weights 하위 그룹의 레이어(그룹) 목록 출력
with h5py.File(h5_path, 'r') as f:
    print("Top-level keys:", list(f.keys()))
    if 'model_weights' in f:
        mw = f['model_weights']
        saved_layer_groups = list(mw.keys())
        print("model_weights 하위 그룹(레이어/서브모듈) 개수:", len(saved_layer_groups))
        print("처음 100개(또는 전체):")
        for name in saved_layer_groups[:100]:
            print(name)
    else:
        print("model_weights 키가 없습니다. (예상과 다른 형식)")

# 2) 예시로 현재 로컬에서 생성되는 모델 레이어 이름(숫자 포함) 확인
# -> 실제 네가 사용하는 model 생성 코드로 대체해야 정확함
model = MobileNet(input_shape=(224,224,3), include_top=True, weights=None, classes=1000)
model_layer_names = [l.name for l in model.layers]
print("현재 코드가 생성하는 모델 레이어 수:", len(model_layer_names))
print("모델 레이어(처음 100):")
for n in model_layer_names[:100]:
    print(n)

# 3) 저장된 그룹 이름과 모델 레이어 이름의 차이점(간단 비교)
# 저장된 그룹들은 보통 'module/block_1/conv...' 형태일 수 있으니 단순 집합 비교
saved_set = set(saved_layer_groups)
model_set = set(model_layer_names)

only_in_saved = sorted(list(saved_set - model_set))
only_in_model = sorted(list(model_set - saved_set))

print("\n저장된 가중치에만 있는 그룹(몇 개):", len(only_in_saved))
print(only_in_saved[:50])
print("\n모델에만 있는 레이어(몇 개):", len(only_in_model))
print(only_in_model[:50])
