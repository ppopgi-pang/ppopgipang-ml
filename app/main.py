from fastapi import FastAPI
from app.core.config import settings
from app.core.containers import Container
from app.api.controllers import vision_controller

def create_app() -> FastAPI:
    container = Container()
    
    # Initialize implementation for DI
    # We can pre-instantiate the vision service to load the model on startup
    # by accessing the provider:
    vision_service = container.vision_service() 
    # This triggers init -> YoloModel.load() -> singleton creation
    
    app = FastAPI(title=settings.PROJECT_NAME)
    app.container = container
    
    app.include_router(vision_controller.router, prefix=f"{settings.API_V1_STR}/vision", tags=["vision"])
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
