# SRT 직렬화 유틸리티
# - ClassifiedSegment 리스트를 SubRip(.srt) 포맷으로 변환
# - 각 블록: 번호 / 타임스탬프 / final_text / 빈줄

from typing import Iterable, List
from schemas.emotions import ClassifiedSegment
from utils.timecode import ms_to_srt_timestamp

def segments_to_srt(segs: Iterable[ClassifiedSegment]) -> str:
    """
    SubRip(.srt) 포맷으로 직렬화
    예시:
    1
    00:00:01,000 --> 00:00:02,000
    (기쁨) 안녕하세요, 모두.
    """
    lines: List[str] = []
    for idx, s in enumerate(segs, start=1):
        # start, end는 초 단위이므로 밀리초로 변환
        start_ms = int(s.start * 1000)
        end_ms = int(s.end * 1000)
        start_tc = ms_to_srt_timestamp(start_ms)
        end_tc = ms_to_srt_timestamp(end_ms)
        lines.append(str(idx))
        lines.append(f"{start_tc} --> {end_tc}")
        lines.append(s.final_text)  # 감정 프리픽스가 붙은 문장
        lines.append("")            # 블록 간 빈 줄
    return "\n".join(lines)
