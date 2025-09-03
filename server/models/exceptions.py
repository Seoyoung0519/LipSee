# server/models/exceptions.py
# ------------------------------------------------------------
# LipSee 서버 모델 예외 처리
# ------------------------------------------------------------

"""
AV-ASR Models Custom Exceptions

현재 시스템에 맞춘 모델 처리 중 발생할 수 있는 예외들을 정의합니다.
"""

class ModelError(Exception):
    """모델 관련 기본 예외 클래스"""
    pass

class KoreanASRError(ModelError):
    """한국어 ASR 처리 중 발생하는 예외"""
    pass

class AVHubertError(ModelError):
    """AV-HuBERT 처리 중 발생하는 예외"""
    pass

class LipAnalysisError(ModelError):
    """입모양 분석 중 발생하는 예외"""
    pass

class FusionError(ModelError):
    """오디오-비디오 특징 융합 중 발생하는 예외"""
    pass

class ModelLoadError(ModelError):
    """모델 로딩 중 발생하는 예외"""
    pass

class InferenceError(ModelError):
    """모델 추론 중 발생하는 예외"""
    pass

class VideoProcessingError(ModelError):
    """비디오 처리 중 발생하는 예외"""
    pass

class ConfigurationError(ModelError):
    """설정 관련 예외"""
    pass

class InputValidationError(ModelError):
    """입력 데이터 검증 실패 예외"""
    pass

class FeatureExtractionError(ModelError):
    """특징 추출 실패 예외"""
    pass