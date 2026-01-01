from pydantic import BaseModel

class DetectRequest(BaseModel):
    context: str | None = None
