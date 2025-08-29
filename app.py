# app.py
import os, time, logging, torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routes.correct import router as ec_router, get_corrector  # get_corrector를 가져와 웜업에 사용

# 런타임 환경(무료/저사양 안정화)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)

log = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 기동 시 1회 로드 + 웜업 (첫 요청 지연 제거)
    t0 = time.time()
    corr = get_corrector()  # KoBARTCorrector 싱글톤 로드
    with torch.no_grad():
        _ = corr.model.generate(**corr.tok("테스트", return_tensors="pt"),
                                max_new_tokens=4, num_beams=1, do_sample=False)
    log.info(f"[startup] EC model/tokenizer ready in {time.time()-t0:.2f}s")
    yield
    # 종료 시 정리 필요하면 여기서

app = FastAPI(title="EC KoBART Service", lifespan=lifespan)
app.include_router(ec_router)

# 루트 접근은 문서로 리다이렉트(게이트웨이가 / 먼저 칠 때 502/404 예방)
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# Render Health Check 용
@app.get("/v2/health")
def v2_health():
    return {"ok": True}

# 기존 호환 헬스
@app.get("/healthz")
def healthz():
    return {"ok": True}
