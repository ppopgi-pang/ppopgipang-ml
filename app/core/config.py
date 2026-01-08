from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "ppopgipang-ml"
    API_V1_STR: str = "/api/v1"

    # YOLO
    YOLO_MODEL_PATH: str = "models/yolo_best.pt"

    # BERT
    BERT_MODEL_PATH: str = "models/bert-finetuned"
    NUM_LABELS: int = 3
    LABEL_MAP: dict[int, str] = {
        0: "class_0",
        1: "class_1",
        2: "class_2",
    }

    class Config:
        env_file = ".env"


settings = Settings()
