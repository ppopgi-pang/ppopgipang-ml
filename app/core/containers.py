from dependency_injector import containers, providers
from app.services.vision_service import VisionService
from app.services.bert_service import BertService

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.api.controllers.vision_controller",
            "app.api.controllers.bert_controller",
        ]
    )

    vision_service = providers.Factory(
        VisionService,
    )

    bert_service = providers.Factory(
        BertService,
    )
