from dependency_injector import containers, providers
from services import SearchService

class Container(containers.DeclarativeContainer):

    wiring_config = containers.WiringConfiguration(modules=["main"])

    search_service = providers.Factory(SearchService)
