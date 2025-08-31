# 시간 유틸리티
# - ms(정수) → SRT 타임코드 "HH:MM:SS,mmm"

def ms_to_srt_timestamp(ms: int) -> str:
    """
    ms 단위를 SRT 타임코드 포맷(HH:MM:SS,mmm)으로 변환
    """
    total_seconds = ms // 1000
    milli = ms % 1000
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{milli:03d}"