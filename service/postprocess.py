# service/postprocess.py
import re

_sent_split = re.compile(r'(?<=[.!?])\s+')

def sent_split(s: str):
    return [p.strip() for p in _sent_split.split(s.strip()) if p.strip()]

def light_normalize(s: str) -> str:
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'([.!?])\1{1,}', r'\1', s)
    s = s.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return s.strip()

def prune_repeated_ngrams(text: str, n: int = 4, max_repeats: int = 1) -> str:
    words = text.split()
    if len(words) < n * (max_repeats + 1):
        return text
    seen = {}
    keep_idx = len(words)
    for i in range(len(words) - n + 1):
        ng = tuple(words[i:i+n])
        seen[ng] = seen.get(ng, 0) + 1
        if seen[ng] > max_repeats:
            keep_idx = i
            break
    return " ".join(words[:keep_idx]).strip()

def postprocess_text(pred: str, n_src_sent: int | None = None, max_chars: int = 160, **kwargs) -> str:
    """
    하위호환: 이전 호출부가 n_src=... 를 넘겨도 동작하도록 처리.
    - n_src_sent: 입력 문장(원문)의 문장 수 추정값
    - max_chars: 최종 출력 최대 길이(문자)
    """
    # <- 중요: 예전 호출부 호환
    if n_src_sent is None:
        n_src_sent = kwargs.pop("n_src", None)
    if n_src_sent is None:
        n_src_sent = 1
    try:
        n_src_sent = max(1, int(n_src_sent))
    except Exception:
        n_src_sent = 1

    # 1) 가벼운 정규화
    pred = light_normalize(pred)

    # 2) 문장 단위 중복 제거
    parts = sent_split(pred)
    dedup = []
    for st in parts:
        if not dedup or st != dedup[-1]:
            dedup.append(st)

    # 3) 문장 수 상한(입력+1, 최소2, 최대3)
    cap = min(max(n_src_sent + 1, 2), 3)
    keep, total = [], 0
    for st in dedup:
        if total + len(st) > max_chars:
            break
        keep.append(st)
        total += len(st)
        if len(keep) >= cap:
            break
    out = " ".join(keep) if keep else pred

    # 4) n-gram 반복 컷 + 마지막 정리
    out = prune_repeated_ngrams(out, n=4, max_repeats=1)
    return light_normalize(out)
