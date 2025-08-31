# 라우터 정의
# - POST /classify : JSON으로 감정 분류 결과 반환
# - POST /srt      : 감정 라벨이 포함된 SRT 본문 반환(다운로드 헤더 포함)

from fastapi import APIRouter, Response, Query
from fastapi.responses import PlainTextResponse
from enum import Enum
from schemas.emotions import ClassifyRequest, ClassifyResponse, SrtRequest
from services.onnx_emotion_classifier import classify_segments
from services.srt_exporter import segments_to_srt
from core.config import settings

router = APIRouter(tags=["emotions"])

class ResponseFormat(str, Enum):
    file = "file"
    json = "json"

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="EC 모델 출력 구조로 감정 라벨링"
)
def classify(req: ClassifyRequest):
    """
    EC 모델 출력 구조(segments[].corrected)를 입력으로 받아 11감정 라벨을 부여합니다.
    - id/start/end 사용 (초 단위)
    - original, picked_candidate, gain, corrected 포함
    - final_text = (emotion) corrected
    """
    # 직접 감정 분류 수행
    enriched = classify_segments(req.segments, add_prefix=req.final_prefix_emotion)
    
    return ClassifyResponse(
        request_id=req.request_id,
        model_version=req.model_version,
        segments=enriched,
        num_segments=len(enriched)
    )

@router.post(
    "/srt",
    responses={
        200: {
            "description": "SRT 파일 다운로드 또는 JSON 응답",
            "content": {
                "application/x-subrip": {"description": "SRT 파일 다운로드"},
                "application/json": {"description": "JSON 응답 (SRT 텍스트 포함)"}
            }
        }
    },
    summary="감정 라벨링 SRT 생성 또는 JSON 응답"
)
def export_srt(
    req: SrtRequest,
    format: ResponseFormat = Query(
        default=ResponseFormat.file, 
        description="응답 형식: 'file'은 SRT 다운로드, 'json'은 JSON 응답"
    )
):
    """
    format 파라미터로 응답 형식 선택:
    - file: SRT 파일 다운로드 (기본값)
    - json: JSON 응답 (SRT 텍스트 포함)
    """
    # 직접 감정 분류 수행
    enriched = classify_segments(req.segments, add_prefix=req.final_prefix_emotion)
    
    if format == ResponseFormat.json:
        # JSON 응답
        return {
            "request_id": req.request_id,
            "model_version": req.model_version,
            "segments": enriched,
            "srt_text": segments_to_srt(enriched),
            "filename": req.filename or "emotion_subtitles.srt"
        }
    else:
        # 파일 다운로드
        srt_text = segments_to_srt(enriched)
        filename = (req.filename or settings.srt_default_filename).replace("\n", "").strip()
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=srt_text, media_type="application/x-subrip", headers=headers)