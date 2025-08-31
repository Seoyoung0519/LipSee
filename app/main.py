# app/main.py
import os
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PipelineResponse
from .utils import load_config, fetch_to_bytes
from .clients import ServiceClients

app = FastAPI(title="LipSee Conductor", version="1.0.1")  # 이름도 추천안 중 하나로 반영

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

CONFIG_PATH = os.getenv("LIPSEE_CONFIG", "config/servers.json")
CONFIG: Dict[str, Any] = load_config(CONFIG_PATH)
clients = ServiceClients(CONFIG)

@app.on_event("startup")
async def _warmup_on_startup():
    await clients.warmup_all()

@app.on_event("shutdown")
async def _shutdown():
    await clients.close()

@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": app.version}

@app.post("/v1/warmup")
async def manual_warmup():
    await clients.warmup_all()
    return {"warmed": True}

@app.post("/v1/pipeline/process", response_model=PipelineResponse)
async def process_pipeline(
    file: Optional[UploadFile] = File(default=None),
    video_url: Optional[str] = Form(default=None),

    language: str = Form(default="ko"),
    diarize: bool = Form(default=False),
    return_words: bool = Form(default=False),
    chunk_ms: int = Form(default=400),
    beam_size: int = Form(default=5),
    fusion_mode: Optional[str] = Form(default=None),
    alpha: Optional[float] = Form(default=None),
    confidence_threshold: Optional[float] = Form(default=None),

    # Emotion 옵션
    final_prefix_emotion: Optional[bool] = Form(default=True),
):
    """
    1) AV-ASR /v1/enhanced_infer
    2) EC /ec/correct
    3) Emotion /classify (+ /srt)
    """
    if not file and not video_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'video_url' is required.")

    # 요청 직전 웜업 시도(슬립 상태 대비)
    await clients.warmup_all()

    # ---- 1) AV-ASR ----
    asr_params = {
        "language": language,
        "format": "json",  # EC로 넘기기 위해 json 고정
        "diarize": str(diarize).lower(),
        "return_words": str(return_words).lower(),
        "chunk_ms": str(chunk_ms),
        "beam_size": str(beam_size),
        "fusion_mode": fusion_mode,
        "alpha": alpha,
        "confidence_threshold": confidence_threshold,
    }
    asr_params = {k: v for k, v in asr_params.items() if v is not None}

    try:
        if file:
            content = await file.read()
            filename = file.filename or "input.mp4"
            content_type = file.content_type or "application/octet-stream"
        else:
            limit_mb = int(CONFIG.get("http", {}).get("download_limit_mb", 800))
            content, content_type = await fetch_to_bytes(
                video_url,
                timeout=int(CONFIG.get("http", {}).get("timeout_seconds", 180)),
                limit_mb=limit_mb
            )
            filename = "video_from_url"

        asr_json = await clients.av_asr_enhanced_infer_file((filename, content, content_type), asr_params)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AV-ASR failed: {e}")

    # ---- 2) EC ----
    try:
        ec_json = await clients.ec_correct(asr_json)
        # sanity: EC 필수 필드가 있는지 가볍게 확인 (오케스트레이터 수준에서 soft-check)
        _segs = ec_json.get("segments", [])
        if _segs and any("corrected" not in s for s in _segs):
            # 일부 구현에서 corrected 대신 text를 돌려줄 수도 있어, 이 경우 emotion 서버 기대 형식에 맞춰 alias
            for s in _segs:
                s.setdefault("text", s.get("corrected", s.get("picked_candidate", s.get("original", ""))))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"EC correction failed: {e}")

    # ---- 3) Emotion ----
    try:
        final_json = await clients.emotion_classify(ec_json)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Emotion classification failed: {e}")

    # SRT: 우선 EC 결과로 시도, 실패 시 분류 결과 JSON으로 폴백
    srt_filename = CONFIG["emotion"]["defaults"].get("srt_filename", "meeting_emotion.srt")
    try:
        srt_out = await clients.emotion_srt(
            ec_json,
            srt_format="file",  # ✅ 고정: file
            filename=srt_filename,
            final_prefix_emotion=final_prefix_emotion
        )
    except Exception:
        try:
            srt_out = await clients.emotion_srt(
                final_json,
                srt_format="file",
                filename=srt_filename,
                final_prefix_emotion=final_prefix_emotion
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"SRT generation failed: {e}")

    # ✅ 항상 SRT 파일 스트림 반환
    return Response(
        content=srt_out,
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{srt_filename}"'}
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
# app/clients.py
from typing import Dict, Any, Optional, Tuple, Union
import httpx
from .utils import retry_request


class ServiceClients:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        http_cfg = config.get("http", {})
        self.timeout = int(http_cfg.get("timeout_seconds", 180))
        self.retry_max = int(http_cfg.get("retry_max", 3))
        self.retry_backoff = float(http_cfg.get("retry_backoff_seconds", 2.0))

        self.av_asr = config["av_asr"]
        self.ec = config["ec"]
        self.emotion = config["emotion"]

        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        await self.client.aclose()

    # ---------- Warmup ----------
    async def warmup_all(self):
        from .utils import try_warmup
        await try_warmup(self.client, self.av_asr["base_url"], self.av_asr.get("warmup"))
        await try_warmup(self.client, self.ec["base_url"], self.ec.get("warmup"))
        await try_warmup(self.client, self.emotion["base_url"], self.emotion.get("warmup"))

    # ---------- AV-ASR ----------
    async def av_asr_enhanced_infer_file(
        self, file_tuple: Tuple[str, bytes, str], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = self.av_asr["base_url"].rstrip("/") + "/v1/enhanced_infer"

        async def _call():
            r = await self.client.post(
                url,
                files={"file": file_tuple},
                data={k: v for k, v in params.items() if v is not None},
            )
            r.raise_for_status()
            return r.json()

        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)

    # ---------- EC ----------
    async def ec_correct(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력: AV-ASR 결과 JSON
        출력: 교정된 segments (original, corrected, picked_candidate, gain 포함)
        """
        url = self.ec["base_url"].rstrip("/") + "/ec/correct"

        async def _call():
            r = await self.client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)

    # ---------- Emotion ----------
    async def emotion_classify(
        self, payload: Dict[str, Any], *, final_prefix_emotion: Optional[bool] = True
    ) -> Dict[str, Any]:
        """
        입력: EC 출력 JSON
        출력: segments[].emotion, score, final_text 추가
        """
        url = self.emotion["base_url"].rstrip("/") + "/classify"
        data = dict(payload)
        if final_prefix_emotion is not None:
            data["final_prefix_emotion"] = bool(final_prefix_emotion)

        async def _call():
            r = await self.client.post(url, json=data)
            r.raise_for_status()
            return r.json()

        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)

    async def emotion_srt(
        self,
        payload: Dict[str, Any],
        *,
        srt_format: str = "file",
        filename: Optional[str] = None,
        final_prefix_emotion: Optional[bool] = True,
    ) -> Union[bytes, Dict[str, Any]]:
        """
        POST /srt
        - file 모드 → bytes (SRT 바이너리)
        - json 모드 → dict (JSON 내 srt_text 포함)
        """
        url = self.emotion["base_url"].rstrip("/") + "/srt"
        data = dict(payload)
        data["format"] = srt_format
        if filename:
            data["filename"] = filename
        if final_prefix_emotion is not None:
            data["final_prefix_emotion"] = bool(final_prefix_emotion)

        async def _call():
            r = await self.client.post(url, json=data)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            if srt_format == "file" and not ctype.startswith("application/json"):
                return r.content
            return r.json()

        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)
