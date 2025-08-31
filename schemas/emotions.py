# 데이터 스키마 정의 (Pydantic v2)
# - 입력: EC 모델 출력 구조 (새로운 형태)
# - 출력: ClassifiedSegment (감정 라벨 및 final_text 포함)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

# 새로운 입력 형태를 위한 스키마
class ECRefinedSegment(BaseModel):
    # EC 모델 출력 구조에 맞춘 스키마
    id: str = Field(..., description="세그먼트 ID")
    start: float = Field(..., ge=0, description="시작 시간 (초)")
    end: float = Field(..., ge=0, description="종료 시간 (초)")
    original: str = Field(..., min_length=1, description="원본 텍스트")
    picked_candidate: Optional[str] = Field(default=None, description="선택된 후보")
    gain: Optional[float] = Field(default=None, description="개선도 점수")
    corrected: str = Field(..., min_length=1, description="교정된 텍스트 (감정 분류에 사용)")

    @field_validator("end")
    @classmethod
    def check_time(cls, v, info):
        # end는 start보다 커야 합니다.
        start = info.data.get("start", 0)
        if v <= start:
            raise ValueError("end must be greater than start")
        return v

class ClassifyRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    # 새로운 입력 형태를 위한 요청 스키마
    request_id: str = Field(..., description="요청 ID")
    model_version: Optional[Dict[str, Any]] = Field(default=None, description="모델 버전 정보")
    segments: List[ECRefinedSegment] = Field(..., min_length=1, description="정교화 세그먼트 리스트")
    # final_text 구성 옵션 (기본: (감정) 프리픽스 부착)
    final_prefix_emotion: Optional[bool] = Field(
        default=True, description="final_text 앞에 (감정) 프리픽스 부착 (기본값: true)"
    )

class ClassifiedSegment(BaseModel):
    # 입력의 기본 메타는 유지
    id: str
    start: float
    end: float
    original: str
    picked_candidate: Optional[str] = None
    gain: Optional[float] = None
    corrected: str
    # 모델 출력
    emotion: str            # 11개 감정 레이블 중 하나 (예: 기쁨/슬픔/분노/…)
    score: float            # softmax 확신도 (0~1)
    final_text: str         # (감정) corrected

class ClassifyResponse(BaseModel):
    request_id: str
    model_version: Optional[Dict[str, Any]] = None
    segments: List[ClassifiedSegment]
    num_segments: int

class SrtRequest(ClassifyRequest):
    # SRT 파일명 지정 가능 (미지정 시 기본값 사용)
    filename: Optional[str] = None

class SrtResponse(BaseModel):
    filename: str
    bytes_len: int
