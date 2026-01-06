from app.models.yolo_model import YoloModel
from app.api.dto.detect_response import DetectResponse, DetectionDto
import io
from PIL import Image
import numpy as np

class VisionService:

    def __init__(self):

        self.model = YoloModel.load()
        self.CONFIDENCE_THRESHOLD = 0.2

    async def detect(self, image_file) -> DetectResponse:
        image_bytes = await image_file.read()
        
        image = Image.open(io.BytesIO(image_bytes))

        results = self.model.detect(image)
        
        return self._to_detect_response(results)

    def _to_detect_response(self, results) -> DetectResponse:

        detections = []
        top_detection = None
        max_conf = -1.0

        for result in results:

            for box in result.boxes:
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
