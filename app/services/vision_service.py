from app.models.yolo_model import YoloModel
from app.api.dto.detect_response import DetectResponse, DetectionDto
import io
from PIL import Image
import numpy as np

class VisionService:

    def __init__(self):
        # In a real DI scenario, we might inject the model factory or instance, 
        # but the prompt specifically asked for YoloModel.load() pattern in __init__ or similar.
        # However, to perfectly follow DI, we could inject it. 
        # But the prompt's example showed `self.model = YoloModel.load()`.
        # I will stick to the prompt's suggested pattern for the model loading to ensure singleton behavior 
        # and cold start handling as requested.
        self.model = YoloModel.load()
        # Configuration for labels, threshold etc can be trusted to be here or in config
        self.CONFIDENCE_THRESHOLD = 0.5 

    async def detect(self, image_file) -> DetectResponse:
        # Read image to bytes
        image_bytes = await image_file.read()
        
        # Convert to format suitable for YOLO if needed, or pass bytes directly if YOLO supports it.
        # Ultralytics supports PIL images, numpy arrays, file paths. 
        # Passing bytes directly might not work with raw YOLO call depending on version, 
        # usually better to convert to PIL Image.
        image = Image.open(io.BytesIO(image_bytes))

        results = self.model.detect(image)
        
        return self._to_detect_response(results)

    def _to_detect_response(self, results) -> DetectResponse:
        # Parse results
        # results is a list of Results objects
        
        detections = []
        top_detection = None
        max_conf = -1.0

        for result in results:
            # boxes object
            for box in result.boxes:
                # box.xyxy, box.conf, box.cls
                try:
                    conf = float(box.conf[0])
                except Exception:
                    conf = 0.0
                    
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue

                try:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                except Exception:
                    label = "unknown"
                
                # Bbox
                bbox = box.xyxy[0].tolist() 
                
                detection = DetectionDto(
                    label=label,
                    displayName=label, # Map if needed
                    confidence=conf,
                    bbox=bbox
                )
                detections.append(detection)
                
                if conf > max_conf:
                    max_conf = conf
                    top_detection = detection

        return DetectResponse(
            topDetection=top_detection,
            detections=detections,
            meta={
                "count": len(detections),
                "model": "yolo"
            }
        )
