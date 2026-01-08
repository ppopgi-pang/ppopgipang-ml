from fastapi import APIRouter, Depends, Body
from dependency_injector.wiring import inject, Provide
from app.api.dto.bert_response import (
    BertClassificationResponse,
    BertBatchClassificationRequest,
)
from app.services.bert_service import BertService
from app.core.containers import Container

router = APIRouter()


@router.post("/classify", response_model=BertClassificationResponse)
@inject
async def classify_text(
    text: str = Body(..., embed=True),
    bert_service: BertService = Depends(Provide[Container.bert_service]),
):
    """
    단일 텍스트 분류
    """
    return await bert_service.classify(text)


@router.post("/classify/batch", response_model=list[BertClassificationResponse])
@inject
async def batch_classify(
    request: BertBatchClassificationRequest,
    bert_service: BertService = Depends(Provide[Container.bert_service]),
):
    """
    배치 텍스트 분류
    """
    return await bert_service.batch_classify(request.texts)
