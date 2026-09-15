from fastapi import FastAPI

from incidentiq.api.routes import router

app = FastAPI(
    title="IncidentIQ",
    description="AI-powered incident investigation system",
    version="0.1.0",
)

app.include_router(router)