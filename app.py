"""
FastAPI 앱 엔트리
- /v2/health: 간단 상태 확인(디바이스/모델경로)
- /v2/correct: EC 서비스 메인 엔드포인트
"""

from fastapi import FastAPI
from routes.correct import router as ec_router
import torch
from config import KOBART_EC_PATH

# 서비스 메타정보
app = FastAPI(title="EC (KoBART) Service", version="2.0.0")

# 라우터 등록
app.include_router(ec_router, prefix="/v2")

@app.get("/v2/health")
def health():
    """
    상태 점검용 엔드포인트:
    - 배포 환경에서 readiness/liveness probe에 활용 가능(상태 모니터링에 사용)
    """
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": KOBART_EC_PATH
    }