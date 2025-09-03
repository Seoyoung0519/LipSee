# server/models/whisper_encoder.py
"""
Whisper Large V3 Audio Encoder for Enhanced AV-ASR System

Whisper Large V3 오디오 인코더
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
import logging

try:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .config import config
    from .exceptions import ModelLoadError, InferenceError
except ImportError:
    # Render 환경에서 절대 import 시도
    from server.models.config import config
    from server.models.exceptions import ModelLoadError, InferenceError

logger = logging.getLogger(__name__)


class WhisperEncoder:
    """Whisper Large V3 오디오 인코더"""
    
    def __init__(
        self,
        model: Optional[WhisperForConditionalGeneration] = None,
        processor: Optional[WhisperProcessor] = None,
        device: str = "cpu"
    ):
        if not TORCH_AVAILABLE:
            raise ModelLoadError("PyTorch and transformers are required for Whisper")
        
        self.model = model
        self.processor = processor
        self.device = device
        self.output_dim = 1024
        
        if model is not None:
            self.model.to(device)
            self.model.eval()
            logger.info(f"Whisper model initialized on {device}")
    
    @staticmethod
    def load(
        model_name: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ) -> "WhisperEncoder":
        try:
            if model_name is None:
                model_name = config.korean_asr.whisper_model_name
            
            # Render 환경에서 캐시 디렉토리 설정
            if cache_dir is None:
                cache_dir = os.getenv("TRANSFORMERS_CACHE", "/tmp/transformers_cache")
                os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Loading Whisper model: {model_name}")
            logger.info(f"Cache directory: {cache_dir}")
            
            processor = WhisperProcessor.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False  # 온라인에서 다운로드 허용
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False  # 온라인에서 다운로드 허용
            )
            
            logger.info("Whisper model loaded successfully")
            return WhisperEncoder(model=model, processor=processor, device=device)
            
        except Exception as e:
            error_msg = f"Failed to load Whisper model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Whisper 모델로 실제 텍스트 추론"""
        try:
            if self.model is None or self.processor is None:
                raise InferenceError("Whisper model or processor not loaded")
            
            if audio.ndim != 1 or len(audio) == 0:
                raise InferenceError(f"Invalid audio input: shape={audio.shape}")
            
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            
            # Whisper는 최소 30초(480,000 샘플)를 요구하므로 패딩 추가
            min_samples = 480000  # 30초 * 16000Hz
            if len(audio) < min_samples:
                # 짧은 오디오는 패딩으로 확장
                padding = np.zeros(min_samples - len(audio), dtype=audio.dtype)
                audio = np.concatenate([audio, padding])
                logger.debug(f"Padded audio from {len(audio) - len(padding)} to {len(audio)} samples")
            
            # WhisperForConditionalGeneration으로 직접 텍스트 추론
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Whisper 자체 디코더로 텍스트 생성
                generated_ids = self.model.generate(
                    **inputs, 
                    language="ko",
                    task="transcribe",
                    max_length=448,
                    num_beams=5
                )
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 신뢰도 계산 (간단한 방법)
                confidence = 0.85  # Whisper는 신뢰도를 직접 제공하지 않으므로 기본값
                
                logger.debug(f"Whisper transcription: {transcription}")
                logger.debug(f"Whisper confidence: {confidence:.3f}")
                
                return {
                    "text": transcription,
                    "confidence": confidence,
                    "generated_ids": generated_ids.cpu().numpy()
                }
                
        except Exception as e:
            error_msg = f"Whisper transcription failed: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
    
    def generate_nbest(self, audio: np.ndarray, sample_rate: int = 16000, num_beams: int = 5) -> List[Dict[str, Any]]:
        """Whisper 모델로 빠른 n-best 후보 생성 (단순화된 beam search)"""
        try:
            if self.model is None or self.processor is None:
                raise InferenceError("Whisper model or processor not loaded")
            
            if audio.ndim != 1 or len(audio) == 0:
                raise InferenceError(f"Invalid audio input: shape={audio.shape}")
            
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            
            # Whisper는 최소 30초(480,000 샘플)를 요구하므로 패딩 추가
            min_samples = 480000  # 30초 * 16000Hz
            if len(audio) < min_samples:
                # 짧은 오디오는 패딩으로 확장
                padding = np.zeros(min_samples - len(audio), dtype=audio.dtype)
                audio = np.concatenate([audio, padding])
                logger.debug(f"Padded audio from {len(audio) - len(padding)} to {len(audio)} samples")
            
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 간단한 beam search로 n-best 후보 생성 (빠른 버전)
                beam_results = self.model.generate(
                    **inputs,
                    language="ko",
                    task="transcribe",
                    num_beams=3,  # beam 수 줄임
                    num_return_sequences=3,  # 반환 수 줄임
                    return_dict_in_generate=True,
                    output_scores=True,
                    early_stopping=True,
                    max_length=448
                )
                
                nbest_candidates = []
                for i, (sequence, score) in enumerate(zip(beam_results.sequences, beam_results.sequences_scores)):
                    # 시퀀스를 텍스트로 디코딩
                    text = self.processor.batch_decode([sequence], skip_special_tokens=True)[0]
                    
                    # 신뢰도 계산 (log score를 확률로 변환)
                    confidence = torch.exp(score).item()
                    
                    # 토큰별 정보 생성
                    tokens = self._create_tokens_from_text(text, confidence)
                    
                    nbest_candidates.append({
                        "rank": i + 1,
                        "text": text,
                        "score": score.item(),
                        "confidence": confidence,
                        "tokens": tokens
                    })
                
                # 점수 순으로 정렬 (높은 점수부터)
                nbest_candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # 기본 결과도 추가
                base_result = self.transcribe(audio)
                base_candidate = {
                    "rank": len(nbest_candidates) + 1,
                    "text": base_result['text'],
                    "score": -1.0,
                    "confidence": base_result['confidence'],
                    "tokens": self._create_tokens_from_text(base_result['text'], base_result['confidence'])
                }
                nbest_candidates.append(base_candidate)
                
                # 최대 5개까지만 반환
                nbest_candidates = nbest_candidates[:5]
                
                logger.debug(f"Whisper generated {len(nbest_candidates)} n-best candidates")
                return nbest_candidates
                
        except Exception as e:
            error_msg = f"Whisper n-best generation failed: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
    
    def _create_tokens_from_text(self, text: str, confidence: float) -> List[Dict[str, Any]]:
        """텍스트로부터 토큰 정보 생성"""
        tokens = []
        words = text.split()
        current_time = 0.0
        
        for word in words:
            token = {
                "text": word,
                "start": current_time,
                "end": current_time + 0.5,
                "logprob": -0.15,
                "confidence": confidence
            }
            tokens.append(token)
            current_time += 0.5
        
        return tokens
    
    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """특징 벡터 추출 (호환성을 위해 유지)"""
        try:
            if self.model is None or self.processor is None:
                raise InferenceError("Whisper model or processor not loaded")
            
            if audio.ndim != 1 or len(audio) == 0:
                raise InferenceError(f"Invalid audio input: shape={audio.shape}")
            
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            
            # Whisper는 최소 30초(480,000 샘플)를 요구하므로 패딩 추가
            min_samples = 480000  # 30초 * 16000Hz
            if len(audio) < min_samples:
                # 짧은 오디오는 패딩으로 확장
                padding = np.zeros(min_samples - len(audio), dtype=audio.dtype)
                audio = np.concatenate([audio, padding])
                logger.debug(f"Padded audio from {len(audio) - len(padding)} to {len(audio)} samples")
            
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # WhisperForConditionalGeneration에서 특징 벡터 추출
                outputs = self.model.model.encoder(**inputs)
                features = outputs.last_hidden_state
                features = features.cpu().numpy()[0]
                
                logger.debug(f"Generated Whisper features: {features.shape}")
                return features.astype(np.float32)
                
        except Exception as e:
            error_msg = f"Whisper encoding failed: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
    
    def _resample_audio(self, audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
        try:
            if src_rate == target_rate:
                return audio
            
            if src_rate > target_rate:
                ratio = src_rate // target_rate
                resampled = audio[::ratio]
            else:
                ratio = target_rate // src_rate
                resampled = np.repeat(audio, ratio)
            
            return resampled
            
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, using original audio")
            return audio
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "whisper_large_v3",
            "model_name": config.korean_asr.whisper_model_name,
            "output_dim": self.output_dim,
            "sample_rate": 16000,
            "frame_hop_ms": 20,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "model_loaded": self.model is not None
        }
