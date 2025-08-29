from fastapi import FastAPI
from routes.correct import router as ec_router

app = FastAPI(title="EC KoBART Service")
app.include_router(ec_router)

# 헬스체크
@app.get("/healthz")
def healthz():
    return {"ok": True}