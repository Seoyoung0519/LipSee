# FastAPI 애플리케이션 엔트리포인트
# - /health: 헬스체크
# - /classify, /srt: routers/emotions.py 에서 라우팅

from fastapi import FastAPI
from routers.emotions import router as emotions_router

app = FastAPI(
    title="LipSee Emotion Classification API",
    version="2.0.0",
    description="EC corrected 자막 기반 11-감정 분류 + SRT 생성"
)

@app.get("/health")
def health():
    """
    단순 헬스 체크 엔드포인트.
    - 배포/모니터링 환경에서 상태 점검용으로 사용
    """
    return {"status": "ok", "service": "emotion-kcelectra", "version": "2.0.0"}

# 요청하신 대로 루트 경로에 바인딩 (prefix 없음)
# => POST /classify, POST /srt
app.include_router(emotions_router)
