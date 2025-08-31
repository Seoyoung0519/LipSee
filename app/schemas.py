# app/schemas.py
from typing import Optional, Dict, Any
from pydantic import BaseModel


class PipelineResponse(BaseModel):
    """
    참고: 현재 /pipeline/process 엔드포인트는 SRT 파일을 반환하므로
    이 모델은 디버그/JSON 엔드포인트용으로만 사용 가능
    """
    final_json: Dict[str, Any]
    srt_text: Optional[str] = None
    asr_json: Dict[str, Any]
    ec_json: Dict[str, Any]
    meta: Dict[str, Any]
