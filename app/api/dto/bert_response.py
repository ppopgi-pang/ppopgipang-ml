from pydantic import BaseModel
from typing import Dict, Any, List


class BertClassificationResponse(BaseModel):
    text: str
    predicted_label: str
    predicted_class: int
    confidence: float
    class_probabilities: Dict[str, float]
    meta: Dict[str, Any]


class BertBatchClassificationRequest(BaseModel):
    texts: List[str]
