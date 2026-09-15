from fastapi import APIRouter
from pydantic import BaseModel
from incidentiq.pipeline import IncidentIQ

DATA_PATH = r"E:\\incidentiq\\data\\processed\\logs.parquet"

incidentiq = IncidentIQ(
    data_path=DATA_PATH
)

router = APIRouter()

class InvestigationRequest(BaseModel):
    query:str

@router.post("/investigate")
def investigate(request: InvestigationRequest):
    return incidentiq.investigate(
        request.query
    )