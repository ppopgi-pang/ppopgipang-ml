from fastapi import APIRouter, UploadFile, File, Depends
from dependency_injector.wiring import inject, Provide
from app.api.dto.detect_response import DetectResponse
from app.services.vision_service import VisionService
from app.core.containers import Container

router = APIRouter()

@router.post("/detect", response_model=DetectResponse)
@inject
async def detect_object(
    image: UploadFile = File(...),
    vision_service: VisionService = Depends(Provide[Container.vision_service])
):
    result = await vision_service.detect(image)
    return result
