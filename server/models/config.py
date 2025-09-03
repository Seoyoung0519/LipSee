# server/models/config.py
# ------------------------------------------------------------
# LipSee 서버 모델 설정
# ------------------------------------------------------------

"""
AV-ASR Models Configuration

현재 시스템에 맞춘 모델 설정들을 중앙에서 관리합니다.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class KoreanASRConfig:
    """한국어 ASR 모델 설정 (Wav2Vec2 + Whisper) - 최고 성능"""
    # Wav2Vec2 설정 (대형 모델 사용)
    wav2vec2_model_name: str = "kresnik/wav2vec2-large-xlsr-korean"  # large 모델로 변경
    wav2vec2_output_dim: int = 1205  # 어휘 크기
    
    # Whisper 설정 (대형 모델 사용)
    whisper_model_name: str = "openai/whisper-large-v3"  # large 모델로 변경
    whisper_output_dim: int = 1205  # 어휘 크기
    
    # 앙상블 설정
    ensemble_method: str = "longer_text"  # "longer_text", "confidence", "weighted"
    audio_weight: float = 0.7  # 오디오 가중치
    
    # 오디오 처리 설정
    sample_rate: int = 16000
    target_fps: int = 25  # 특징 벡터 프레임 레이트

@dataclass
class AVHubertConfig:
    """AV-HuBERT + 입모양 분석 모델 설정 (팀원 구현)"""
    # AV-HuBERT 설정
    output_dim: int = 512  # AV-HuBERT 출력 차원
    input_height: int = 112
    input_width: int = 112
    input_channels: int = 1  # grayscale
    default_model_name: str = "avhubert_base.onnx"
    
    # 입모양 분석 모델 설정
    lip_analysis_model_name: str = "lip_analysis_model.onnx"
    lip_analysis_output_dim: int = 256  # BiGRU 출력 차원
    
    # 비디오 처리 설정
    target_fps: int = 25  # 비디오 프레임 레이트
    max_frames: int = 300  # 최대 프레임 수
    
    # 더미 모드 설정
    allow_dummy: bool = True  # 더미 모드 허용 (기본값: True)
    use_freeze_avhubert: bool = True  # AV-HuBERT freeze 사용

@dataclass
class FusionConfig:
    """오디오-비디오 특징 융합 설정"""
    # 융합 방식
    fusion_method: str = "weighted"  # "weighted", "concatenate", "attention"
    
    # 가중 융합 설정
    audio_weight: float = 0.7  # 오디오 가중치
    video_weight: float = 0.3  # 비디오 가중치
    
    # 차원 변환 설정
    enable_dimension_matching: bool = True  # 차원 맞춤 활성화
    use_linear_interpolation: bool = True  # 선형 보간 사용

@dataclass
class ModelConfig:
    """전체 모델 설정"""
    # 한국어 ASR 설정
    korean_asr: KoreanASRConfig = field(default_factory=KoreanASRConfig)
    
    # AV-HuBERT + 입모양 분석 설정
    avhubert: AVHubertConfig = field(default_factory=AVHubertConfig)
    
    # 융합 설정
    fusion: FusionConfig = field(default_factory=FusionConfig)
    
    # 환경변수에서 설정 로드
    def __post_init__(self):
        # 한국어 ASR 설정
        if os.getenv("WAV2VEC2_MODEL_NAME"):
            self.korean_asr.wav2vec2_model_name = os.getenv("WAV2VEC2_MODEL_NAME")
        if os.getenv("WHISPER_MODEL_NAME"):
            self.korean_asr.whisper_model_name = os.getenv("WHISPER_MODEL_NAME")
        if os.getenv("AUDIO_WEIGHT"):
            self.korean_asr.audio_weight = float(os.getenv("AUDIO_WEIGHT"))
        
        # AV-HuBERT 설정
        if os.getenv("AVHUBERT_OUTPUT_DIM"):
            self.avhubert.output_dim = int(os.getenv("AVHUBERT_OUTPUT_DIM"))
        if os.getenv("AVHUBERT_MODEL_NAME"):
            self.avhubert.default_model_name = os.getenv("AVHUBERT_MODEL_NAME")
        
        # 융합 설정
        if os.getenv("FUSION_METHOD"):
            self.fusion.fusion_method = os.getenv("FUSION_METHOD")
        if os.getenv("AUDIO_WEIGHT"):
            self.fusion.audio_weight = float(os.getenv("AUDIO_WEIGHT"))
            self.fusion.video_weight = 1.0 - self.fusion.audio_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "korean_asr": {
                "wav2vec2_model": self.korean_asr.wav2vec2_model_name,
                "whisper_model": self.korean_asr.whisper_model_name,
                "ensemble_method": self.korean_asr.ensemble_method,
                "audio_weight": self.korean_asr.audio_weight,
                "target_fps": self.korean_asr.target_fps
            },
            "avhubert": {
                "model_name": self.avhubert.default_model_name,
                "output_dim": self.avhubert.output_dim,
                "input_shape": f"[T, {self.avhubert.input_height}, {self.avhubert.input_width}]",
                "target_fps": self.avhubert.target_fps
            },
            "fusion": {
                "method": self.fusion.fusion_method,
                "audio_weight": self.fusion.audio_weight,
                "video_weight": self.fusion.video_weight,
                "dimension_matching": self.fusion.enable_dimension_matching
            }
        }

# 전역 설정 인스턴스
config = ModelConfig()