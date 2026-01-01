from pydantic import BaseModel
from typing import List, Dict, Any

class DetectionDto(BaseModel):
    label: str
    displayName: str
    confidence: float
    bbox: List[float]

class DetectResponse(BaseModel):
    topDetection: DetectionDto | None
    detections: List[DetectionDto]
    meta: Dict[str, Any]
