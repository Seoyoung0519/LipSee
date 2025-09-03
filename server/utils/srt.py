# server/utils/srt.py
"""
SRT 자막 변환 유틸리티
"""

from typing import List, Dict, Any


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    세그먼트 리스트를 SRT 형식으로 변환
    
    Args:
        segments: 세그먼트 리스트 [{"start": float, "end": float, "text": str}, ...]
    
    Returns:
        SRT 형식의 문자열
    """
    srt_content = []
    
    for i, segment in enumerate(segments, 1):
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")  # 빈 줄
    
    return "\n".join(srt_content)


def format_time(seconds: float) -> str:
    """
    초를 SRT 시간 형식으로 변환
    
    Args:
        seconds: 초 단위 시간
    
    Returns:
        SRT 시간 형식 (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
