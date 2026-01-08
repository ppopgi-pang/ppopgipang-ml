from fastapi import FastAPI
from app.core.config import settings
from app.core.containers import Container
from app.api.controllers import vision_controller, bert_controller

def create_app() -> FastAPI:
    container = Container()
    
    print("Initializing services and loading models...")
    container.vision_service()
    container.bert_service()
    print("All models loaded successfully!")
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="YOLO와 BERT를 포함한 딥러닝 모델 서빙 API",
    )
    app.container = container
    
    app.include_router(
        vision_controller.router,
        prefix=f"{settings.API_V1_STR}/vision",
        tags=["vision"],
    )
    app.include_router(
        bert_controller.router,
        prefix=f"{settings.API_V1_STR}/bert",
        tags=["bert"],
    )

    @app.get("/")
    async def root():
        return {
            "message": "ppopgipang-ml API",
            "endpoints": {
                "vision": f"{settings.API_V1_STR}/vision/detect",
                "bert": f"{settings.API_V1_STR}/bert/classify",
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
