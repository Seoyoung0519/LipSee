"""
합본/분할 및 마이크로배칭 유틸
- 여러 세그먼트를 SEG 토큰으로 연결해 한 번에 인퍼런스(문맥 확보)
- 인퍼런스 결과를 SEG 기준으로 다시 분할
- 문자수 기준 마이크로배칭으로 과도한 길이 방지
"""
from typing import List
from config import SEG_TOKEN, CHUNK_CHAR_LIMIT

def join_with_marker(texts: List[str]) -> str:
    """
    빈문장도 슬롯 유지(인덱스 보존) 위해 그대로 합침
    예: ["A", "B", ""] -> "A <SEG> B <SEG> "
    """
    return f" {SEG_TOKEN} ".join(t if t else "" for t in texts)

def split_by_marker(full_text: str) -> List[str]:
    """
    KoBART 출력에서 SEG 기준 재분할
    trailing 구분자 등으로 빈 조각이 나올 수 있어 strip만 적용
    """
    parts = [p.strip() for p in full_text.split(SEG_TOKEN)]
    return parts

def microbatch_texts(texts: List[str]) -> List[List[str]]:
    """
    간단한 문자수 기준 마이크로배칭
    - 너무 긴 합본은 모델 입력 한도 및 지연 증가를 유발하므로 분할
    """
    batches, buf, cur_len = [], [], 0
    for t in texts:
        tlen = len(t) if t else 0
        if buf and (cur_len + tlen + 10 > CHUNK_CHAR_LIMIT):
            batches.append(buf)
            buf, cur_len = [], 0
        buf.append(t)
        cur_len += tlen
    if buf:
        batches.append(buf)
    return batches