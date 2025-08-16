"""
보수적 후처리
- 공백 정리, 문장부호 보강
- 의미 손상 방지를 위해 약어/토큰 화이트리스트를 간단 보정
"""
import re

# 의미 보존을 위한 간단 화이트리스트(필요 시 확장)
ALLOWLIST = {"AI","GPU","API","HTTP","URL","ms","MB","GB","KoBART"}

def gentle_cleanup(src: str, dst: str) -> str:
    # 약어/영문 토큰이 원문에 있는데 출력에 사라졌다면 간단 복구
    for token in ALLOWLIST:
        if token in src and token not in dst:
            dst = dst.replace(token.lower(), token).replace(token.capitalize(), token)
    # 연속 공백 정리
    dst = re.sub(r"\s+", " ", dst).strip()
    # 문장 끝 문장부호 보강(너무 짧은 텍스트는 제외)
    if (not dst.endswith(("?", "!", "."))) and len(dst) > 1:
        dst += "."
    return dst