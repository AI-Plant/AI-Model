import base64

def image_to_base64(image_path):
    """
    지정된 경로의 이미지 파일을 Base64 문자열로 인코딩하여 반환합니다.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        return "오류: 파일을 찾을 수 없습니다."
    except Exception as e:
        return f"인코딩 오류: {e}"

image_file_path = "C:\smp\SmPlant\AI-Model\saved_model\mobilenet_plant_classifier.h5"
base64_data = image_to_base64(image_file_path)

if base64_data.startswith("오류"):
    print(base64_data)
else:
    print(base64_data)
    