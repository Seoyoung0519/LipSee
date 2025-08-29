# schemas/asr_payload.py
from typing import List, Any, Optional
from pydantic import BaseModel, Field

class NBestItem(BaseModel):
    text: str
    score: Optional[float] = None

class Word(BaseModel):
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None
    logprob: Optional[float] = None

class Segment(BaseModel):
    id: str
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    no_speech_prob: Optional[float] = None
    frame_entropy: Optional[float] = None
    tokens: Optional[List[Any]] = None
    words: Optional[List[Word]] = None
    nbest: Optional[List[NBestItem]] = None

class ASRPayload(BaseModel):
    request_id: str
    model_version: dict | None = None
    media: dict | None = None
    encoders: dict | None = None
    decoder: dict | None = None
    segments: List[Segment]
    hotwords: List[str] = Field(default_factory=list)
    domain_lexicon: List[str] = Field(default_factory=list)
