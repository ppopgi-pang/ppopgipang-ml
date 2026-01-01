from ultralytics import YOLO
from app.core.config import settings
import os

class YoloModel:
    _instance = None

    @classmethod
    def load(cls):
        if cls._instance is None:
            # Ensure directory exists or handle path correctness
            # For now assuming settings.MODEL_PATH is correct relative to CWD or absolute
            print(f"Loading YOLO model from {settings.MODEL_PATH}...")
            # If model file strictly needed, we might want to check existence, 
            # but ultralytics might download if not found (for standard models), 
            # or fail for custom path. 
            # The prompt suggested "models/best.pt".
            cls._instance = YOLO(settings.MODEL_PATH) 
        return cls._instance

    def detect(self, image):
        # image can be bytes, PIL Image, numpy array, etc.
        # Ultralytics handles various formats.
        return self._instance(image)
