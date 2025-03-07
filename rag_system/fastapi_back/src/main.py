from typing import Annotated, Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.api.v1.api_route import router
from src.logger import main_logger


app = FastAPI(
    title="ml_app",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    openapi_tags=[
        {
            "name": "ml_app",
            "description": "Сервис для обучения и тестирования tf-idf",
        },
    ]
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Функция логирование запросов"""
    main_logger.info("Incoming request: %s %s", request.method, request.url)
    response = await call_next(request)
    main_logger.info("Outgoing response: Status %d", response.status_code)
    return response


class StatusResponse(BaseModel):
    status: Annotated[str, Literal["App healthy", "App unavailable"]]


def check_service_status() -> StatusResponse:
    """Функция проверки статуса сервера"""
    service_available = True
    status = "App healthy" if service_available else "App unavailable"
    return StatusResponse(status=status)


@app.get("/", response_model=StatusResponse)
async def root() -> StatusResponse:
    """Центральная функция запуска основного ip"""
    main_logger.info("Root endpoint called")
    return check_service_status()


# Роутер с префиксом /api/v1/models
app.include_router(router)

if __name__ == "__main__":
    pass
