from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ppopgipang-vision"
    API_V1_STR: str = "/api/v1"
    MODEL_PATH: str = "models/best.pt"
    
    class Config:
        env_file = ".env"

settings = Settings()
