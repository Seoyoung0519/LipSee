# schemas/ec.py
from typing import List
from pydantic import BaseModel

class ECResultSeg(BaseModel):
    id: str
    start: float
    end: float
    original: str
    corrected: str

class ECResponse(BaseModel):
    request_id: str
    model: str = "kobart_ec"
    version: str = "local"
    segments: List[ECResultSeg]
