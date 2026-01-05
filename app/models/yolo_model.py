from ultralytics import YOLO
from app.core.config import settings
import os

class YoloModel:
    _instance = None
    _model = None

    @classmethod
    def load(cls):
        if cls._instance is None:
            cls._instance = cls()
            print(f"Loading YOLO model from {settings.MODEL_PATH}...")
            cls._instance._model = YOLO(settings.MODEL_PATH)
        return cls._instance

    def detect(self, image):
        # image can be bytes, PIL Image, numpy array, etc.
        # Ultralytics handles various formats.
        return self._model(image)
