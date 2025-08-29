# service/phonetic.py
# 자모 분해 + 레벤슈타인 거리 (정규화)
import unicodedata

CHO = [ chr(x) for x in range(0x1100, 0x1113) ]
JUNG = [ chr(x) for x in range(0x1161, 0x1176) ]
JONG = [ chr(x) for x in range(0x11A8, 0x11C3) ] + [""]

def decompose(s: str) -> str:
    out=[]
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            base = code - 0xAC00
            cho = base // 588
            jung = (base % 588)//28
            jong = base % 28 - 1
            out.append(CHO[cho]); out.append(JUNG[jung]); 
            if jong>=0: out.append(JONG[jong])
        else:
            out.append(ch)
    return "".join(out)

def lev(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la==0 and lb==0: return 0.0
    dp = list(range(lb+1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = min(
                dp[j] + 1,
                dp[j-1] + 1,
                prev + (0 if ca==cb else 1)
            )
            prev, dp[j] = dp[j], cur
    return dp[lb] / max(la, lb)

def phonetic_distance(a: str, b: str) -> float:
    return lev(decompose(a), decompose(b))

def gen_candidates(span_text: str, vocab: list[str], K=5, max_dist=0.28):
    scored = []
    for v in set(vocab):
        d = phonetic_distance(span_text, v)
        if d <= max_dist:
            scored.append((d, v))
    scored.sort(key=lambda x: x[0])
    return [v for _,v in scored[:K]]
