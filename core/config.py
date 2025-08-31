# 환경설정 모듈
# - 모델 ID, SRT 기본 파일명 등 환경변수로 덮어쓰기 가능

from pydantic import BaseModel
import os

class Settings(BaseModel):
    # HuggingFace 모델 ID (기본: 11감정 KcELECTRA)
    model_id: str = os.getenv(
        "MODEL_ID",
        "nlp04/korean_sentiment_analysis_kcelectra"
    )
    # SRT 기본 파일명
    srt_default_filename: str = os.getenv("SRT_DEFAULT_FILENAME", "emotion_subtitles.srt")
    # 토크나이저 최대 길이 (긴 문장에 대한 truncation 길이)
    max_length: int = int(os.getenv("MAX_LENGTH", "256"))
    # (확장용) 배치 인퍼런스 사용 여부
    use_batch: bool = os.getenv("USE_BATCH", "false").lower() == "true"

settings = Settings()
