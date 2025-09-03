# server/models/wav2vec2_encoder.py
"""
Wav2Vec2 Korean Audio Encoder for Enhanced AV-ASR System

한국어 특화 Wav2Vec2 오디오 인코더
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
import logging

try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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


class Wav2Vec2Encoder:
    """Wav2Vec2 한국어 오디오 인코더"""
    
    def __init__(
        self,
        model: Optional[Wav2Vec2ForCTC] = None,
        processor: Optional[Wav2Vec2Processor] = None,
        device: str = "cpu"
    ):
        if not TORCH_AVAILABLE:
            raise ModelLoadError("PyTorch and transformers are required for Wav2Vec2")
        
        self.model = model
        self.processor = processor
        self.device = device
        self.output_dim = 768
        
        if model is not None:
            self.model.to(device)
            self.model.eval()
            logger.info(f"Wav2Vec2 model initialized on {device}")
    
    @staticmethod
    def load(
        model_name: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ) -> "Wav2Vec2Encoder":
        try:
            if model_name is None:
                model_name = config.korean_asr.wav2vec2_model_name
            
            # Render 환경에서 캐시 디렉토리 설정
            if cache_dir is None:
                cache_dir = os.getenv("TRANSFORMERS_CACHE", "/tmp/transformers_cache")
                os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Loading Wav2Vec2 model: {model_name}")
            logger.info(f"Cache directory: {cache_dir}")
            
            processor = Wav2Vec2Processor.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False  # 온라인에서 다운로드 허용
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False  # 온라인에서 다운로드 허용
            )
            
            logger.info("Wav2Vec2 model loaded successfully")
            return Wav2Vec2Encoder(model=model, processor=processor, device=device)
            
        except Exception as e:
            error_msg = f"Failed to load Wav2Vec2 model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Wav2Vec2 모델로 실제 텍스트 추론"""
        try:
            if self.model is None or self.processor is None:
                raise InferenceError("Wav2Vec2 model or processor not loaded")
            
            if audio.ndim != 1 or len(audio) == 0:
                raise InferenceError(f"Invalid audio input: shape={audio.shape}")
            
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            
            # Wav2Vec2ForCTC로 직접 텍스트 추론
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # CTC 디코딩으로 텍스트 생성
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                
                # 신뢰도 계산
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                confidence = torch.max(probabilities).item()
                
                logger.debug(f"Wav2Vec2 transcription: {transcription}")
                logger.debug(f"Wav2Vec2 confidence: {confidence:.3f}")
                
                return {
                    "text": transcription,
                    "confidence": confidence,
                    "logits": logits.cpu().numpy()
                }
                
        except Exception as e:
            error_msg = f"Wav2Vec2 transcription failed: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
    
    def generate_nbest(self, audio: np.ndarray, sample_rate: int = 16000, num_beams: int = 5) -> List[Dict[str, Any]]:
        """Wav2Vec2 모델로 빠른 n-best 후보 생성 (단어 조합 기반)"""
        try:
            if self.model is None or self.processor is None:
                raise InferenceError("Wav2Vec2 model or processor not loaded")
            
            if audio.ndim != 1 or len(audio) == 0:
                raise InferenceError(f"Invalid audio input: shape={audio.shape}")
            
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 기본 결과 생성
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                base_text = self.processor.batch_decode(predicted_ids)[0]
                base_confidence = torch.max(torch.nn.functional.softmax(logits, dim=-1)).item()
                
                nbest_candidates = []
                
                # 1. 기본 결과
                nbest_candidates.append({
                    "rank": 1,
                    "text": base_text,
                    "score": -1.0,
                    "confidence": base_confidence,
                    "tokens": self._create_tokens_from_text(base_text, base_confidence)
                })
                
                # 2. 단어 조합 기반 빠른 후보 생성
                words = base_text.split()
                if len(words) >= 2:
                    # 단어 순서 변경 (인접한 단어들만)
                    for i in range(min(2, len(words) - 1)):
                        alt_words = words.copy()
                        alt_words[i], alt_words[i+1] = alt_words[i+1], alt_words[i]
                        alt_text = " ".join(alt_words)
                        
                        if alt_text != base_text:
                            nbest_candidates.append({
                                "rank": len(nbest_candidates) + 1,
                                "text": alt_text,
                                "score": -2.0 - i,
                                "confidence": base_confidence * 0.9,
                                "tokens": self._create_tokens_from_text(alt_text, base_confidence * 0.9)
                            })
                
                # 3. 간단한 변형 후보들
                variations = [
                    base_text.replace("하겠습니다", "하겠습니다."),
                    base_text.replace("하겠습니다.", "하겠습니다"),
                    base_text + ".",
                    base_text.rstrip(".")
                ]
                
                for i, variant in enumerate(variations):
                    if variant != base_text and variant not in [c["text"] for c in nbest_candidates]:
                        nbest_candidates.append({
                            "rank": len(nbest_candidates) + 1,
                            "text": variant,
                            "score": -3.0 - i,
                            "confidence": base_confidence * 0.85,
                            "tokens": self._create_tokens_from_text(variant, base_confidence * 0.85)
                        })
                
                # 4. 최대 5개까지만 반환
                nbest_candidates = nbest_candidates[:5]
                
                logger.debug(f"Wav2Vec2 generated {len(nbest_candidates)} n-best candidates")
                return nbest_candidates
                
        except Exception as e:
            error_msg = f"Wav2Vec2 n-best generation failed: {str(e)}"
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
                "end": current_time + 0.4,
                "logprob": -0.1,
                "confidence": confidence
            }
            tokens.append(token)
            current_time += 0.4
        
        return tokens
    
    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """특징 벡터 추출 (호환성을 위해 유지)"""
        try:
            if self.model is None or self.processor is None:
                raise InferenceError("Wav2Vec2 model or processor not loaded")
            
            if audio.ndim != 1 or len(audio) == 0:
                raise InferenceError(f"Invalid audio input: shape={audio.shape}")
            
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Wav2Vec2ForCTC에서 특징 벡터 추출
                outputs = self.model.wav2vec2(**inputs)
                features = outputs.last_hidden_state
                features = features.cpu().numpy()[0]
                
                logger.debug(f"Generated Wav2Vec2 features: {features.shape}")
                return features.astype(np.float32)
                
        except Exception as e:
            error_msg = f"Wav2Vec2 encoding failed: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
    
    def _resample_audio(self, audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
        try:
            if src_rate == target_rate:
                return audio
            
            audio_tensor = torch.from_numpy(audio).float()
            resampler = torchaudio.transforms.Resample(orig_freq=src_rate, new_freq=target_rate)
            resampled = resampler(audio_tensor)
            return resampled.numpy()
            
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, using original audio")
            return audio
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "wav2vec2_korean",
            "model_name": config.korean_asr.wav2vec2_model_name,
            "output_dim": self.output_dim,
            "sample_rate": 16000,
            "frame_hop_ms": 20,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "model_loaded": self.model is not None
        }
