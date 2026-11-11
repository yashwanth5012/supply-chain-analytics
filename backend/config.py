from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_TITLE: str = "Enterprise Supply Chain Analytics Engine"
    API_VERSION: str = "3.0"
    DEFAULT_LEAD_TIME_DAYS: float = 7.0
    DEFAULT_HOLDING_COST: float = 2.5
    DEFAULT_ORDER_COST: float = 50.0
    DEFAULT_SERVICE_LEVEL: float = 0.95

settings = Settings()