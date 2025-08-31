# app/utils.py
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import httpx


def load_config(path: str = "config/servers.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def try_warmup(
    client: httpx.AsyncClient, base_url: str, warmups: Optional[List[Dict[str, str]]] = None
) -> None:
    """
    각 서비스에 대해 지정된 warmup 엔드포인트 호출
    실패해도 무시 (콜드스타트 방지 목적)
    """
    if not warmups:
        return
    tasks = []
    for w in warmups:
        method = (w.get("method") or "GET").upper()
        path = w.get("path") or "/"
        url = base_url.rstrip("/") + path
        tasks.append(_one_warm_call(client, method, url))
    await asyncio.gather(*tasks, return_exceptions=True)


async def _one_warm_call(client: httpx.AsyncClient, method: str, url: str):
    try:
        if method == "GET":
            r = await client.get(url)
        elif method == "HEAD":
            r = await client.head(url)
        else:
            r = await client.request(method, url)
        r.raise_for_status()
    except Exception:
        pass


async def retry_request(call, *, retry_max: int, backoff: float):
    last_exc = None
    for attempt in range(1, retry_max + 1):
        try:
            return await call()
        except Exception as e:
            last_exc = e
            if attempt == retry_max:
                break
            await asyncio.sleep(backoff * attempt)
    raise last_exc


async def fetch_to_bytes(
    url: str, *, timeout: int, limit_mb: int
) -> Tuple[bytes, str]:
    """
    video_url을 바이트로 가져와 AV-ASR에 파일로 넘길 수 있게 변환
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        content = r.content
        size_mb = len(content) / (1024 * 1024)
        if size_mb > limit_mb:
            raise ValueError(
                f"Downloaded file too large: {size_mb:.1f} MB (limit {limit_mb} MB)"
            )
        ctype = r.headers.get("Content-Type", "application/octet-stream")
        return content, ctype
