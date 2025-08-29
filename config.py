# config.py
from __future__ import annotations
import os, json
from types import SimpleNamespace as NS
from typing import Dict, Any

# 기본값 (config_ec.json에서 일부만 제공돼도 안전하게 동작)
DEFAULTS: Dict[str, Any] = {
    "kobart_model_dir": "models/kobart_ec",
    "decode": {
        "num_beams": 6,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 9,
        "encoder_no_repeat_ngram_size": 9,
        "repetition_penalty": 1.5,
        "length_penalty": 0.9,
        "early_stopping": True,
        "renormalize_logits": True,
        "bad_words": ["!!!!","???","ㅋㅋㅋㅋ","ㅎㅎㅎㅎ"],
        "max_input_tokens": 256,
        "max_new_tokens_floor": 32,
        "max_new_tokens_ceil": 96,
        "max_chars": 160
    },
    "guard": {
        "cer_gate": 0.40,
        "max_sent_cap": 2
    },
    "thresholds": {
        "aggressive_threshold": 0.05,  # tau
        "nbest_margin": 1.0,
        "PHONETIC_MAX": 0.45,
        "GAIN_MIN": 0.50,
        "LOGPROB_LOW": -0.80,
        "CONF_FALLBACK": 0.90,
        "MAX_CAND": 5
    }
}

# config 파일 경로: ENV > 로컬 파일 > 기본값
CONFIG_PATH = os.environ.get(
    "EC_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "config_ec.json")
)

def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _to_ns(d: Any) -> Any:
    if isinstance(d, dict):
        return NS(**{k: _to_ns(v) for k,v in d.items()})
    if isinstance(d, list):
        return [_to_ns(x) for x in d]
    return d

def _load_dict() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
    else:
        user = {}
    return _deep_merge(DEFAULTS, user)

# ---- 로딩 & 공개 객체 ----
_cfg_dict = _load_dict()
cfg = _to_ns(_cfg_dict)  # ✅ 이제 from config import cfg 사용 가능

# ---- 레거시 상수(기존 코드 호환) ----
KOBART_EC_PATH       = cfg.kobart_model_dir
NUM_BEAMS            = cfg.decode.num_beams
NO_REPEAT_NGRAM      = cfg.decode.no_repeat_ngram_size
REPETITION_PENALTY   = cfg.decode.repetition_penalty
LENGTH_PENALTY       = cfg.decode.length_penalty
GUARD_CER            = cfg.guard.cer_gate
GUARD_MAX_SENT       = cfg.guard.max_sent_cap
TAU_GAIN             = cfg.thresholds.aggressive_threshold
NBEST_MARGIN         = cfg.thresholds.nbest_margin
PHONETIC_MAX         = cfg.thresholds.PHONETIC_MAX
GAIN_MIN             = cfg.thresholds.GAIN_MIN
LOGPROB_LOW          = cfg.thresholds.LOGPROB_LOW
CONF_FALLBACK        = cfg.thresholds.CONF_FALLBACK
MAX_CAND             = cfg.thresholds.MAX_CAND

def refresh_cfg(path: str | None = None):
    """config_ec.json 변경 후 런타임 갱신하려면 호출"""
    global CONFIG_PATH, _cfg_dict, cfg
    if path:
        CONFIG_PATH = path
    _cfg_dict = _load_dict()
    cfg = _to_ns(_cfg_dict)
    # 상수들도 갱신이 필요하면 여기에 재할당 로직을 추가하세요.
    return cfg

