# app/clients.py
from typing import Dict, Any, Optional, Tuple
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

    async def warmup_all(self):
        from .utils import try_warmup
        await try_warmup(self.client, self.av_asr["base_url"], self.av_asr.get("warmup"))
        await try_warmup(self.client, self.ec["base_url"], self.ec.get("warmup"))
        await try_warmup(self.client, self.emotion["base_url"], self.emotion.get("warmup"))

    # ---------- Calls ----------
    async def av_asr_enhanced_infer_file(self, file_tuple: Tuple[str, bytes, str], params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.av_asr["base_url"].rstrip("/") + "/v1/enhanced_infer"
        async def _call():
            r = await self.client.post(url, files={"file": file_tuple}, data={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            return r.json()
        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)

    async def ec_correct(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        EC 입력: AV-ASR 전체 JSON (segments[].text, nbest[].text 등 포함)
        EC 출력(예시):
        {
          "segments": [
            {
              "id": "seg_00001",
              "start": 0.0,
              "end": 2.1,
              "original": "...",
              "corrected": "...",
              "picked_candidate": "...",
              "gain": 0.87
            },
            ...
          ]
        }
        """
        url = self.ec["base_url"].rstrip("/") + "/ec/correct"
        async def _call():
            r = await self.client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)

    async def emotion_classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.emotion["base_url"].rstrip("/") + "/classify"
        async def _call():
            r = await self.client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)

    async def emotion_srt(self, payload: Dict[str, Any]) -> str:
        url = self.emotion["base_url"].rstrip("/") + "/srt"
        async def _call():
            r = await self.client.post(url, json=payload)
            r.raise_for_status()
            return r.text
        return await retry_request(_call, retry_max=self.retry_max, backoff=self.retry_backoff)
