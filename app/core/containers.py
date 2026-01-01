from dependency_injector import containers, providers
from app.services.vision_service import VisionService

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["app.api.controllers.vision_controller"])

    vision_service = providers.Factory(
        VisionService,
    )
