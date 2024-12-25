import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from api.v1.api_route import router
from typing import Annotated, Literal

tags_metadata: list[dict[str, str]] = [
    {
        "name": "ml_app",
        "description": "Сервис для обучения и тестирования tf-idf",
    },
]

app = FastAPI(
    title="ml_app",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    openapi_tags=tags_metadata
)


class StatusResponse(BaseModel):
    status: Annotated[str, Literal["App healthy", "App unavailable"]]


def check_service_status() -> StatusResponse:
    service_available = True
    status = "App healthy" if service_available else "App unavailable"
    return StatusResponse(status=status)


@app.get("/", response_model=StatusResponse)
async def root() -> StatusResponse:
    return check_service_status()


# Роутер с префиксом /api/v1/models
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
