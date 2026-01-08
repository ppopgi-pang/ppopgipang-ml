# ppopgipang-ml

YOLO와 BERT 모델을 함께 서빙하는 FastAPI 프로젝트입니다.

## 프로젝트 구조

```
ppopgipang-ml/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── containers.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_model.py
│   │   └── bert_model.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vision_service.py
│   │   └── bert_service.py
│   └── api/
│       ├── __init__.py
│       ├── dto/
│       │   ├── __init__.py
│       │   ├── detect_request.py
│       │   ├── detect_response.py
│       │   └── bert_response.py
│       └── controllers/
│           ├── __init__.py
│           ├── vision_controller.py
│           └── bert_controller.py
├── models/
│   ├── yolo_best.pt
│   └── bert-finetuned/
│       ├── config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── special_tokens_map.json
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 설치 및 실행

1) 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) 의존성 설치

```bash
pip install -r requirements.txt
```

3) 모델 파일 준비

- `models/yolo_best.pt` 파일 배치
- `models/bert-finetuned/` 디렉토리에 BERT 파일들 배치

4) 서버 실행

```bash
uvicorn app.main:app --reload
```

또는

```bash
python -m app.main
```

## 환경 변수

`.env` 파일을 생성하여 경로와 라벨 정보를 설정합니다.

```env
PROJECT_NAME=ppopgipang-ml
API_V1_STR=/api/v1

# YOLO 모델
YOLO_MODEL_PATH=models/yolo_best.pt

# BERT 모델
BERT_MODEL_PATH=models/bert-finetuned
NUM_LABELS=3
```

## API 테스트

### YOLO 객체 탐지

```bash
curl -X POST "http://localhost:8000/api/v1/vision/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg"
```

### BERT 텍스트 분류 (단일)

```bash
curl -X POST "http://localhost:8000/api/v1/bert/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "테스트 텍스트입니다"}'
```

### BERT 텍스트 분류 (배치)

```bash
curl -X POST "http://localhost:8000/api/v1/bert/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "첫 번째 텍스트",
      "두 번째 텍스트",
      "세 번째 텍스트"
    ]
  }'
```

## API 문서

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
