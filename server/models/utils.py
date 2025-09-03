# server/models/utils.py
# ------------------------------------------------------------
# LipSee 서버 모델 유틸리티
# ------------------------------------------------------------

"""
AV-ASR Models Utilities

현재 시스템에 맞춘 모델 관련 유틸리티 함수들을 제공합니다.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    모델 로깅을 설정합니다.
    
    Args:
        level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: 로그 포맷 문자열
        log_file: 로그 파일 경로 (None이면 콘솔만)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 포맷터 생성
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (지정된 경우)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 모델 관련 로거들 설정
    model_loggers = [
        "models.avhubert_encoder",
        "models.lip_analysis_model",
        "models.korean_asr"
    ]
    
    for logger_name in model_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))

def get_model_summary() -> dict:
    """
    현재 시스템의 모델 요약 정보를 반환합니다.
    
    Returns:
        모델 정보 딕셔너리
    """
    try:
        from .config import config
        
        summary = {
            "korean_asr": {
                "available": True,
                "config": {
                    "wav2vec2_model": config.korean_asr.wav2vec2_model_name,
                    "whisper_model": config.korean_asr.whisper_model_name,
                    "ensemble_method": config.korean_asr.ensemble_method,
                    "audio_weight": config.korean_asr.audio_weight,
                    "target_fps": config.korean_asr.target_fps
                }
            },
            "avhubert": {
                "available": True,
                "config": {
                    "model_name": config.avhubert.default_model_name,
                    "output_dim": config.avhubert.output_dim,
                    "input_shape": f"[T, {config.avhubert.input_height}, {config.avhubert.input_width}]",
                    "target_fps": config.avhubert.target_fps
                }
            },
            "fusion": {
                "available": True,
                "config": {
                    "method": config.fusion.fusion_method,
                    "audio_weight": config.fusion.audio_weight,
                    "video_weight": config.fusion.video_weight
                }
            }
        }
        
        return summary
        
    except Exception as e:
        return {
            "error": f"Failed to get model summary: {str(e)}",
            "available": False
        }

def validate_model_config() -> bool:
    """
    현재 시스템의 모델 설정 유효성을 검증합니다.
    
    Returns:
        설정이 유효하면 True, 아니면 False
    """
    try:
        from .config import config
        
        # 한국어 ASR 설정 검증
        if not config.korean_asr.wav2vec2_model_name or not config.korean_asr.whisper_model_name:
            return False
        if config.korean_asr.audio_weight < 0 or config.korean_asr.audio_weight > 1:
            return False
        
        # AV-HuBERT 설정 검증
        if config.avhubert.output_dim <= 0:
            return False
        if config.avhubert.input_height <= 0 or config.avhubert.input_width <= 0:
            return False
        
        # 융합 설정 검증
        if config.fusion.audio_weight < 0 or config.fusion.audio_weight > 1:
            return False
        if abs(config.fusion.audio_weight + config.fusion.video_weight - 1.0) > 1e-6:
            return False
        
        return True
        
    except Exception:
        return False

def check_model_files() -> Dict[str, bool]:
    """
    필요한 모델 파일들의 존재 여부를 확인합니다.
    
    Returns:
        모델 파일별 존재 여부 딕셔너리
    """
    try:
        from .config import config
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        model_files = {
            "avhubert": os.path.join(base_dir, "models", config.avhubert.default_model_name),
            "lip_analysis": os.path.join(base_dir, "models", config.avhubert.lip_analysis_model_name)
        }
        
        existence_status = {}
        for model_name, model_path in model_files.items():
            existence_status[model_name] = os.path.exists(model_path)
        
        return existence_status
        
    except Exception as e:
        return {
            "error": f"Failed to check model files: {str(e)}"
        }