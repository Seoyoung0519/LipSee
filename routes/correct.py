# routes/correct.py
from __future__ import annotations
import logging

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException

from service.kobart_corrector import KoBARTCorrector
from service.rerank import choose_with_gain
from config import cfg  # config_ec.json을 로딩해 둔 객체라고 가정

router = APIRouter(prefix="/ec", tags=["ec"])
logger = logging.getLogger("ec.correct")

# ----- Lazy singleton: KoBARTCorrector 한 번만 로드 -----
_corrector: KoBARTCorrector | None = None


def get_corrector() -> KoBARTCorrector:
    global _corrector
    if _corrector is None:
        # config_ec.json 의 "kobart_model_dir" 사용
        model_dir = getattr(cfg, "kobart_model_dir", "models/kobart_ec")
        _corrector = KoBARTCorrector(model_dir=model_dir)
    return _corrector


@router.post("/correct")
def correct(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    EC 교정 메인 엔드포인트.
    입력: AV-ASR 전체 JSON (segments[].text, nbest[].text 등)
    출력: segments[].{original, corrected, picked_candidate, gain}
    """
    try:
        segments = payload.get("segments", None)
        if not isinstance(segments, list):
            raise HTTPException(status_code=400, detail="payload.segments must be a list")

        # aggressive_threshold(tau) 를 config에서 읽기 (기본 0.05)
        try:
            tau_gain = float(getattr(getattr(cfg, "thresholds", object()), "aggressive_threshold", 0.05))
        except Exception:
            tau_gain = 0.05

        corrector = get_corrector()

        out_segments: List[Dict[str, Any]] = []

        for seg in segments:
            if not isinstance(seg, dict):
                continue

            seg_id = seg.get("id")
            seg_text = seg.get("text") or ""  # ← KeyError/NameError 방지
            nbest = seg.get("nbest") or []

            # --- 후보 수집: nbest에서 원문과 다른 텍스트만 추출 ---
            candidate_texts: List[str] = []
            for hyp in nbest:
                if isinstance(hyp, dict):
                    t = hyp.get("text")
                    if t and t != seg_text:
                        candidate_texts.append(t)
            # 중복 제거(순서 유지)
            candidate_texts = list(dict.fromkeys(candidate_texts))

            # --- 문맥 점수 기반 선택 (gain) ---
            # 시그니처: choose_with_gain(base_text, candidates, tok, model, device, *, tau_gain, tau_len, length_penalty, max_length)
            chosen, picked, gain = choose_with_gain(
                base_text=seg_text,
                candidates=candidate_texts,
                tok=corrector.tok,
                model=corrector.model,
                device=corrector.device,
                tau_gain=tau_gain,     # 임계치(aggressive_threshold)
                tau_len=3,             # 호환용(현재 미사용)
                length_penalty=0.0,
                max_length=256,
            )

            # --- 최종 표면 교정(KoBARTCorrector + GuardRails) ---
            corrected = corrector.correct_batch([chosen])[0]

            out_segments.append({
                "id": seg_id,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "original": seg_text,
                "picked_candidate": picked,  # 채택된 nbest 후보(없으면 null)
                "gain": float(gain),
                "corrected": corrected,
                # 필요시 디버그를 위해 아래 필드도 추가 가능
                # "candidates": candidate_texts,
                # "nbest": nbest,
            })

        return {
            "request_id": payload.get("request_id"),
            "model_version": payload.get("model_version"),
            "segments": out_segments,
        }

    except HTTPException:
        # 이미 의미있는 상태코드 포함 → 그대로 전파
        raise
    except Exception as e:
        # 콘솔에 전체 스택 찍기
        logger.exception("EC /ec/correct failed")
        # 클라이언트에는 짧게
        raise HTTPException(status_code=500, detail=f"/ec/correct failed: {e}")
