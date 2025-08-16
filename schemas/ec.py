"""
입력/출력 JSON 스키마 정의 — Pydantic v2 버전
- 단일 필드 범위 검사는 Field(...)로 처리 (예: ge=0)
- 교차 필드 검증(end_ms >= start_ms)은 @model_validator(mode="after") 사용
- confidence, words는 옵셔널: 들어오면 그대로 응답에 실어주고, 안 오면 무시
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

class WordItem(BaseModel):
    """단어 단위 타임스탬프/신뢰도 (옵션)"""
    start_ms: int = Field(ge=0, description="단어 시작 시각(ms)")
    end_ms: int = Field(ge=0, description="단어 종료 시각(ms)")
    text: str
    confidence: Optional[float] = Field(
        default=None, description="단어 인식 신뢰도(0~1, 옵션)"
    )

    @model_validator(mode="after")
    def _validate_time_order(self):
        """end_ms >= start_ms 보장 (단어 단위)"""
        if self.end_ms < self.start_ms:
            raise ValueError("WordItem.end_ms must be >= WordItem.start_ms")
        return self

class InSegment(BaseModel):
    """
        EC API 입력 세그먼트
        - AV-ASR 세그먼트 그대로 전달받음
        - EC는 text만 교정하고, id/start_ms/end_ms/메타데이터는 그대로 보존
    """
    id: str
    start_ms: int = Field(ge=0, description="세그먼트 시작 시각(ms)")
    end_ms: int = Field(ge=0, description="세그먼트 종료 시각(ms)")
    text: str                     # AV-ASR가 인식한 문장/구
    # --- optional passthrough ---
    confidence: Optional[float] = Field(
        default=None, description="세그먼트 인식 신뢰도(0~1, 옵션)"
    )
    words: Optional[List[WordItem]] = Field(
        default=None, description="단어 단위 타임스탬프 배열(옵션)"
    )

    @model_validator(mode="after")
    # 종료시간이 시작시간보다 빠를 수 없도록 검증
    def _validate_time_order(self):
        """end_ms >= start_ms 보장 (세그먼트 단위)"""
        if self.end_ms < self.start_ms:
            raise ValueError("InSegment.end_ms must be >= InSegment.start_ms")
        return self


class ECRequest(BaseModel):
    """
        EC 요청: AV-ASR가 반환한 segments 배열을 그대로 받는다.
    """
    segments: List[InSegment]

    @field_validator("segments")
    @classmethod
    def _non_empty_segments(cls, v: List[InSegment]):
        """빈 배열 방지(원한다면 완전 허용해도 무방)"""
        # 비어도 동작은 가능하지만, 운영상 흔한 실수라면 막아둘 수 있음.
        # 필요 없으면 이 블록을 삭제하세요.
        if v is None:
            raise ValueError("segments is required")
        return v

class OutSegment(BaseModel):
    """
       EC 응답 세그먼트
       - id/start_ms/end_ms는 반드시 보존(싱크 유지)
       - original: 원문(ASR 결과)
       - corrected: 교정문(EC 결과)
       - confidence/words: 입력에 들어왔으면 그대로 에코백
    """
    id: str
    start_ms: int
    end_ms: int
    original: str     # 원문(ASR 결과)
    corrected: str    # 교정문(EC 결과)
    # --- optional passthrough (그대로 에코백) ---
    confidence: Optional[float] = None
    words: Optional[List[WordItem]] = None

    @model_validator(mode="after")
    def _validate_time_order(self):
        """end_ms >= start_ms 보장 (응답에도 동일 제약 유지)"""
        if self.end_ms < self.start_ms:
            raise ValueError("OutSegment.end_ms must be >= OutSegment.start_ms")
        return self


class ECResponse(BaseModel):
    """
    EC 응답: 세그먼트 배열
    """
    segments: List[OutSegment]