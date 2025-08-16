"""
/v2/correct 엔드포인트
- 입력: AV-ASR segments(JSON)
- 처리: text만 KoBART로 교정
- 출력: 동일한 타임스탬프/ID에 corrected 텍스트 매칭
- confidence/words는 들어오면 그대로 응답에 에코백(passthrough)
"""

from fastapi import APIRouter, HTTPException
from schemas.ec import ECRequest, ECResponse, OutSegment
from service.kobart_corrector import KoBARTCorrector
from service.batching import join_with_marker, split_by_marker, microbatch_texts
from service.postprocess import gentle_cleanup

router = APIRouter()

# 프로세스 시작 시 1회 모델 로드(메모리에 상주시켜 지연 최소화)
_corrector = KoBARTCorrector()

@router.post("/correct", response_model=ECResponse)
def correct(req: ECRequest):
    """
    핵심 보장:
    - 세그먼트의 id/start_ms/end_ms는 절대 변경하지 않음(동일 싱크 보장)
    - EC 결과 길이가 입력과 다르면 원문 유지(시간축 깨짐 방지)
    """
    try:
        segs = req.segments or []
        if not segs:
            return {"segments": []}

        # 1) 원문 텍스트만 추출
        texts = [s.text or "" for s in segs]

        # 2) 문자수 기준 마이크로배칭으로 합본 인퍼런스
        corrected_texts: list[str] = []
        for batch in microbatch_texts(texts):
            # 합본: "문장1 <SEG> 문장2 <SEG> 문장3"
            joined = join_with_marker(batch)
            # KoBART 인퍼런스(출력도 같은 마커로 이어짐)
            joined_out = _corrector.correct_batch([joined])[0]
            # 다시 SEG 기준으로 분할
            parts = split_by_marker(joined_out)
            # 길이 불일치 시 안전하게 해당 배치 구간 원문 유지
            if len(parts) != len(batch):
                parts = batch
            corrected_texts.extend(parts)

        # 3) 전체 개수 일치 최종 검증(안전장치)
        if len(corrected_texts) != len(texts):
            corrected_texts = texts

        # 4) 보수적 후처리 + 메타데이터 에코 + 타임스탬프/ID 보존
        out_items = []
        for s, fixed in zip(segs, corrected_texts):
            corrected = gentle_cleanup(s.text, fixed) if s.text else s.text
            out_items.append(OutSegment(
                id=s.id,                 # ✅ ID 보존
                start_ms=s.start_ms,     # ✅ 시작 타임스탬프 보존
                end_ms=s.end_ms,         # ✅ 종료 타임스탬프 보존
                original=s.text,         # 원문 기록
                corrected=corrected,     # 교정문 기록
                confidence=s.confidence, # (옵션) 그대로 에코백
                words=s.words            # (옵션) 그대로 에코백
            ))

        return ECResponse(segments=out_items)

    except Exception as e:
        # 내부 에러를 노출하지 않도록 간단 메시지 처리
        raise HTTPException(status_code=500, detail=f"ec_internal_error: {e}")