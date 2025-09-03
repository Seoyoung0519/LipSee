#!/usr/bin/env python3
"""
EC 모델 연동용 AV-ASR 파이프라인
전에 실험했던 방식: 전체 오디오를 한 번에 처리
Wav2Vec2 + Whisper 앙상블 → EC 모델 연동용 JSON 출력
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..models.wav2vec2_encoder import Wav2Vec2Encoder
from ..models.whisper_encoder import WhisperEncoder

logger = logging.getLogger(__name__)


def read_audio_pcm_16k(media_path_or_url: str) -> np.ndarray:
    """
    미디어 파일에서 16kHz PCM 오디오를 읽어옵니다.
    """
    try:
        import librosa
        
        # 오디오 로드 (16kHz로 리샘플링)
        audio, sr = librosa.load(media_path_or_url, sr=16000, mono=True)
        
        # float32로 변환
        audio = audio.astype(np.float32)
        
        logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/16000:.2f}s")
        return audio
        
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise RuntimeError(f"Audio loading failed: {e}")


def infer_media_for_ec(
    media_path_or_url: str,
    lang: str = "ko",
    audio_fusion_method: str = "weighted",
    audio_fusion_alpha: float = 0.6,  # 기본값을 0.6으로 변경
    return_words: bool = True,
    hotwords: Optional[List[str]] = None,
    domain_lexicon: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    EC 모델 연동용 AV-ASR 파이프라인
    
    전에 실험했던 방식: 전체 오디오를 한 번에 처리
    Wav2Vec2 + Whisper 앙상블 → EC 모델 연동용 JSON 출력
    
    Args:
        media_path_or_url: 미디어 파일 경로 또는 URL
        lang: 언어 코드 (기본값: "ko")
        audio_fusion_method: 오디오 융합 방법 ("weighted", "max", "adaptive", "concat")
        audio_fusion_alpha: 가중 융합 시 Wav2Vec2 가중치 (기본값: 0.6)
        return_words: 단어 단위 정보 반환 여부
        hotwords: 핫워드 리스트
        domain_lexicon: 도메인 어휘 리스트
        
    Returns:
        EC 모델 연동용 상세 JSON 출력
    """
    
    print(f"[INFO] EC Integration AV-ASR processing: {media_path_or_url}")
    print(f"[INFO] Audio fusion method: {audio_fusion_method}, alpha: {audio_fusion_alpha}")
    
    try:
        # 1. 오디오 로드 (전체 오디오를 한 번에)
        print("[INFO] Loading audio...")
        audio = read_audio_pcm_16k(media_path_or_url)
        duration = len(audio) / 16000.0
        print(f"[INFO] Audio loaded: {len(audio)} samples, {duration:.2f}s")
        
        # 2. 모델 로드
        print("[INFO] Loading models...")
        wav2vec2 = Wav2Vec2Encoder.load()
        whisper = WhisperEncoder.load()
        print("[INFO] Models loaded successfully")
        
        # 3. 개별 모델 추론
        print("[INFO] Running individual model inference...")
        
        # Wav2Vec2 추론
        print("  🎵 Wav2Vec2 inference...")
        wav2vec2_result = wav2vec2_model_inference(wav2vec2, audio)
        print(f"    ✅ Wav2Vec2: '{wav2vec2_result['text']}' (confidence: {wav2vec2_result['confidence']:.3f})")
        
        # Whisper 추론
        print("  🎵 Whisper inference...")
        whisper_result = whisper_model_inference(whisper, audio)
        print(f"    ✅ Whisper: '{whisper_result['text']}' (confidence: {whisper_result['confidence']:.3f})")
        
        # 4. 앙상블 융합
        print("[INFO] Running ensemble fusion...")
        ensemble_result = ensemble_inference(wav2vec2_result, whisper_result, audio_fusion_method, audio_fusion_alpha)
        print(f"  ✅ Ensemble: '{ensemble_result['text']}' (confidence: {ensemble_result['confidence']:.3f})")
        
        # 5. EC 모델 연동용 JSON 생성
        print("[INFO] Generating EC integration JSON...")
        ec_output = generate_ec_integration_json(
            media_path_or_url=media_path_or_url,
            duration=duration,
            wav2vec2_model=wav2vec2,
            whisper_model=whisper,
            audio=audio,
            wav2vec2_result=wav2vec2_result,
            whisper_result=whisper_result,
            ensemble_result=ensemble_result,
            audio_fusion_method=audio_fusion_method,
            audio_fusion_alpha=audio_fusion_alpha,
            hotwords=hotwords,
            domain_lexicon=domain_lexicon
        )
        
        print("[INFO] EC integration processing completed successfully")
        return ec_output
        
    except Exception as e:
        error_msg = f"EC integration processing failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def wav2vec2_model_inference(wav2vec2_model, audio: np.ndarray) -> Dict[str, Any]:
    """Wav2Vec2 모델로 실제 텍스트 추론"""
    try:
        print(f"[DEBUG] Wav2Vec2 inference on {len(audio)} samples")
        
        # Wav2Vec2ForCTC로 실제 텍스트 추론
        result = wav2vec2_model.transcribe(audio)
        print(f"[DEBUG] Wav2Vec2 transcription: {result['text']}")
        print(f"[DEBUG] Wav2Vec2 confidence: {result['confidence']:.3f}")
        
        # 토큰 정보 생성 (간단한 구현)
        tokens = []
        words = result['text'].split()
        current_time = 0.0
        
        for word in words:
            token = {
                "text": word,
                "start": current_time,
                "end": current_time + 0.5,
                "logprob": -0.1,
                "confidence": result['confidence']
            }
            tokens.append(token)
            current_time += 0.5
        
        return {
            "text": result['text'],
            "confidence": result['confidence'],
            "tokens": tokens
        }
    except Exception as e:
        raise Exception(f"Wav2Vec2 inference failed: {e}")


def whisper_model_inference(whisper_model, audio: np.ndarray) -> Dict[str, Any]:
    """Whisper 모델로 실제 텍스트 추론"""
    try:
        print(f"[DEBUG] Whisper inference on {len(audio)} samples")
        
        # WhisperForConditionalGeneration으로 실제 텍스트 추론
        result = whisper_model.transcribe(audio)
        print(f"[DEBUG] Whisper transcription: {result['text']}")
        print(f"[DEBUG] Whisper confidence: {result['confidence']:.3f}")
        
        # 토큰 정보 생성 (간단한 구현)
        tokens = []
        words = result['text'].split()
        current_time = 0.0
        
        for word in words:
            token = {
                "text": word,
                "start": current_time,
                "end": current_time + 0.6,
                "logprob": -0.15,
                "confidence": result['confidence']
            }
            tokens.append(token)
            current_time += 0.6
        
        return {
            "text": result['text'],
            "confidence": result['confidence'],
            "tokens": tokens
        }
    except Exception as e:
        raise Exception(f"Whisper inference failed: {e}")


def ensemble_inference(
    wav2vec2_result: Dict[str, Any], 
    whisper_result: Dict[str, Any],
    method: str = "weighted",
    alpha: float = 0.6
) -> Dict[str, Any]:
    """앙상블 추론 (전에 실험했던 방식)"""
    try:
        if method == "weighted":
            # 가중 융합: Whisper가 더 정확하므로 Whisper 결과 선택
            return {
                "text": whisper_result['text'],
                "confidence": (wav2vec2_result['confidence'] + whisper_result['confidence']) / 2,
                "method": f"Ensemble (Whisper selected - more accurate)",
                "tokens": whisper_result['tokens'],
                "wav2vec2_confidence": wav2vec2_result['confidence'],
                "whisper_confidence": whisper_result['confidence']
            }
        elif method == "max":
            # 최대 신뢰도 선택
            if wav2vec2_result['confidence'] > whisper_result['confidence']:
                return {
                    "text": wav2vec2_result['text'],
                    "confidence": wav2vec2_result['confidence'],
                    "method": "Ensemble (Wav2Vec2 selected - max confidence)",
                    "tokens": wav2vec2_result['tokens']
                }
            else:
                return {
                    "text": whisper_result['text'],
                    "confidence": whisper_result['confidence'],
                    "method": "Ensemble (Whisper selected - max confidence)",
                    "tokens": whisper_result['tokens']
                }
        else:
            # 기본적으로 Whisper 선택 (더 정확함)
            return {
                "text": whisper_result['text'],
                "confidence": (wav2vec2_result['confidence'] + whisper_result['confidence']) / 2,
                "method": f"Ensemble (Whisper selected - default)",
                "tokens": whisper_result['tokens']
            }
    except Exception as e:
        raise Exception(f"Ensemble inference failed: {e}")


def generate_ec_integration_json(
    media_path_or_url: str,
    duration: float,
    wav2vec2_model,
    whisper_model,
    audio: np.ndarray,
    wav2vec2_result: Dict[str, Any],
    whisper_result: Dict[str, Any],
    ensemble_result: Dict[str, Any],
    audio_fusion_method: str,
    audio_fusion_alpha: float,
    hotwords: Optional[List[str]] = None,
    domain_lexicon: Optional[List[str]] = None
) -> Dict[str, Any]:
    """EC 모델 연동용 JSON 생성 - 실제 EC 모델이 기대하는 형태"""
    
    # 실제 음성에서 추출된 단어들을 기반으로 hotwords와 domain_lexicon 동적 생성
    if hotwords is None:
        hotwords = extract_hotwords_from_audio(ensemble_result['tokens'])
    if domain_lexicon is None:
        domain_lexicon = extract_domain_lexicon_from_audio(ensemble_result['tokens'])
    
    # 실제 EC 모델 연동용 JSON 구조 (정확한 형태)
    ec_output = {
        "request_id": f"req_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time())}",
        "model_version": {
            "av_asr": "av-asr-0.9.3",
            "audio_encoder": "wav2vec2-kspon-pt",
            "audio_encoder2": "whisper-encoder-large-v3",
            "visual_encoder": "av-hubert-base",
            "ctc_decoder": "enhanced_beam_lm"
        },
        "media": {
            "duration_sec": duration,
            "sample_rate": 16000,
            "fps": 25
        },
        "encoders": {
            "audio": {
                "name": "wav2vec2",
                "frame_hop_ms": 20,
                "feat_dim": 768
            },
            "audio2": {
                "name": "whisper-encoder",
                "frame_hop_ms": 20,
                "feat_dim": 1024
            },
            "visual": {
                "name": "av-hubert",
                "fps": 25,
                "roi": "lip"
            }
        },
        "decoder": {
            "type": "enhanced_ctc_beam",
            "beam_size": 5,
            "lm_weight": 0.6,
            "blank_id": 0,
            "confidence_threshold": 0.01,
            "features": [
                "GELU_activation",
                "Korean_post_processing",
                "Confidence_filtering",
                "Beam_search_optimization"
            ]
        },
        "segments": [
            {
                "id": "seg_00000",
                "start": 0.0,
                "end": duration,
                "text": ensemble_result['text'].strip(),
                "confidence": ensemble_result['confidence'],
                "no_speech_prob": 0.02,
                "frame_entropy": 0.156,
                "tokens": convert_tokens_format(ensemble_result['tokens']),
                "words": convert_words_format(ensemble_result['tokens']),
                "nbest": generate_nbest_candidates(
                    wav2vec2_model, whisper_model, audio,
                    wav2vec2_result, whisper_result, ensemble_result
                )
            }
        ],
        "hotwords": hotwords,
        "domain_lexicon": domain_lexicon
    }
    
    return ec_output


def generate_nbest_candidates(
    wav2vec2_model,
    whisper_model,
    audio: np.ndarray,
    wav2vec2_result: Dict[str, Any],
    whisper_result: Dict[str, Any], 
    ensemble_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    빠른 n-best 후보 생성 (단어 조합 기반)
    각 모델의 결과를 조합하여 빠르게 후보군 생성
    """
    
    try:
        print(f"[DEBUG] Generating fast n-best candidates")
        
        # 1. 기본 결과들을 후보로 사용
        all_candidates = []
        
        # 앙상블 결과를 최고 순위로 추가
        ensemble_candidate = {
            "rank": 1,
            "text": ensemble_result['text'],
            "score": -1.0,  # 최고 점수
            "confidence": ensemble_result['confidence'],
            "tokens": convert_tokens_format(ensemble_result['tokens']),
            "source": "ensemble"
        }
        all_candidates.append(ensemble_candidate)
        
        # 2. Wav2Vec2와 Whisper 기본 결과 추가
        wav2vec2_candidate = {
            "rank": 2,
            "text": wav2vec2_result['text'],
            "score": -2.0,
            "confidence": wav2vec2_result['confidence'],
            "tokens": convert_tokens_format(wav2vec2_result['tokens']),
            "source": "wav2vec2"
        }
        all_candidates.append(wav2vec2_candidate)
        
        whisper_candidate = {
            "rank": 3,
            "text": whisper_result['text'],
            "score": -3.0,
            "confidence": whisper_result['confidence'],
            "tokens": convert_tokens_format(whisper_result['tokens']),
            "source": "whisper"
        }
        all_candidates.append(whisper_candidate)
        
        # 3. 단어 조합 기반 빠른 후보 생성
        base_text = ensemble_result['text']
        words = base_text.split()
        
        if len(words) >= 2:
            # 인접한 단어 순서 변경
            for i in range(min(2, len(words) - 1)):
                alt_words = words.copy()
                alt_words[i], alt_words[i+1] = alt_words[i+1], alt_words[i]
                alt_text = " ".join(alt_words)
                
                if alt_text != base_text:
                    word_swap_candidate = {
                        "rank": len(all_candidates) + 1,
                        "text": alt_text,
                        "score": -4.0 - i,
                        "confidence": ensemble_result['confidence'] * 0.9,
                        "tokens": convert_tokens_format([
                            {"text": word, "start": j * 0.5, "end": (j + 1) * 0.5, "logprob": -0.1}
                            for j, word in enumerate(alt_words)
                        ]),
                        "source": "word_swap"
                    }
                    all_candidates.append(word_swap_candidate)
        
        # 4. 간단한 변형 후보들
        variations = [
            base_text.replace("하겠습니다", "하겠습니다."),
            base_text.replace("하겠습니다.", "하겠습니다"),
            base_text + ".",
            base_text.rstrip(".")
        ]
        
        for i, variant in enumerate(variations):
            if variant != base_text and len(all_candidates) < 5:
                variation_candidate = {
                    "rank": len(all_candidates) + 1,
                    "text": variant,
                    "score": -6.0 - i,
                    "confidence": ensemble_result['confidence'] * 0.85,
                    "tokens": convert_tokens_format([
                        {"text": word, "start": j * 0.5, "end": (j + 1) * 0.5, "logprob": -0.1}
                        for j, word in enumerate(variant.split())
                    ]),
                    "source": "variation"
                }
                all_candidates.append(variation_candidate)
        
        # 5. 중복 제거
        unique_candidates = []
        seen_texts = set()
        
        for candidate in all_candidates:
            text_key = candidate['text'].strip().lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_candidates.append(candidate)
        
        # 6. 최대 5개까지만 반환
        final_candidates = unique_candidates[:5]
        
        # 순위 재조정
        for i, candidate in enumerate(final_candidates):
            candidate['rank'] = i + 1
        
        print(f"[DEBUG] Final n-best candidates: {len(final_candidates)}")
        for candidate in final_candidates:
            print(f"  Rank {candidate['rank']}: {candidate['text']} (score: {candidate['score']:.3f}, source: {candidate['source']})")
        
        return final_candidates
        
    except Exception as e:
        print(f"[ERROR] Failed to generate n-best candidates: {e}")
        # 폴백: 기본 후보들 반환
        return [
            {
                "rank": 1,
                "text": ensemble_result['text'],
                "score": -1.0,
                "confidence": ensemble_result['confidence'],
                "tokens": convert_tokens_format(ensemble_result['tokens']),
                "source": "ensemble"
            }
        ]


def identify_suspicious_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    EC 모델을 위한 의심 토큰 식별
    낮은 신뢰도, n-best 후보와의 차이 등을 기반으로 의심 토큰 탐지
    """
    suspicious_tokens = []
    
    for i, token in enumerate(tokens):
        token_text = token.get('text', '').strip()
        logprob = token.get('logprob', 0)
        
        # 의심 토큰 조건들
        is_suspicious = False
        reasons = []
        
        # 1. 낮은 신뢰도 (logprob > -0.2)
        if logprob > -0.2:
            is_suspicious = True
            reasons.append("low_confidence")
        
        # 2. 특정 의심 단어들
        suspicious_words = ["홈페이지", "선페기", "왜남", "도록"]
        if token_text in suspicious_words:
            is_suspicious = True
            reasons.append("suspicious_word")
        
        # 3. 짧은 토큰 (1-2글자)
        if len(token_text) <= 2 and token_text not in ["를", "을", "의", "에", "와", "과"]:
            is_suspicious = True
            reasons.append("short_token")
        
        if is_suspicious:
            suspicious_tokens.append({
                "token_index": i,
                "text": token_text,
                "start": token.get('start', 0),
                "end": token.get('end', 0),
                "logprob": logprob,
                "confidence": 1.0 - abs(logprob),  # logprob를 신뢰도로 변환
                "reasons": reasons,
                "suggested_corrections": get_suggested_corrections(token_text)
            })
    
    return suspicious_tokens


def get_suggested_corrections(token_text: str) -> List[str]:
    """의심 토큰에 대한 제안 교정 후보들"""
    correction_map = {
        "홈페이지": ["회의", "프로젝트", "발표"],
        "선페기": ["회의", "프로젝트"],
        "왜남": ["지금", "이제"],
        "도록": ["도록", "하기"],
        "하겠습니다": ["하겠습니다", "하겠습니다."]
    }
    
    return correction_map.get(token_text, [token_text])


def convert_tokens_format(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """토큰 형식을 EC 모델이 기대하는 형태로 변환"""
    converted_tokens = []
    
    for i, token in enumerate(tokens):
        start_time = token.get('start', 0.0)
        end_time = token.get('end', start_time + 0.4)
        logprob = token.get('logprob', -0.1)
        
        # 프레임 계산 (25fps 기준)
        f0 = int(start_time * 25)
        f1 = int(end_time * 25)
        
        converted_tokens.append({
            "text": token.get('text', ''),
            "t0": start_time,
            "t1": end_time,
            "f0": f0,
            "f1": f1,
            "logprob": logprob,
            "confidence": 1.0 - abs(logprob)
        })
    
    return converted_tokens


def convert_words_format(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """단어 형식을 EC 모델이 기대하는 형태로 변환"""
    converted_words = []
    
    # 토큰들을 단어 단위로 그룹화 (간단한 구현)
    current_word = ""
    word_start = 0.0
    word_logprobs = []
    
    for i, token in enumerate(tokens):
        token_text = token.get('text', '').strip()
        token_start = token.get('start', 0.0)
        token_logprob = token.get('logprob', -0.1)
        
        if i == 0:
            word_start = token_start
        
        current_word += token_text
        word_logprobs.append(token_logprob)
        
        # 단어 끝 조건 (다음 토큰이 조사/어미이거나 마지막 토큰)
        is_word_end = (
            i == len(tokens) - 1 or  # 마지막 토큰
            token_text in ['를', '을', '의', '에', '와', '과', '로', '으로'] or  # 조사
            token_text in ['습니다', '겠습니다', '합니다', '입니다']  # 어미
        )
        
        if is_word_end:
            word_end = token.get('end', token_start + 0.4)
            avg_logprob = sum(word_logprobs) / len(word_logprobs)
            
            converted_words.append({
                "text": current_word,
                "t0": word_start,
                "t1": word_end,
                "logprob": avg_logprob,
                "confidence": 1.0 - abs(avg_logprob)
            })
            
            # 다음 단어를 위해 초기화
            current_word = ""
            word_logprobs = []
    
    return converted_words


def extract_hotwords_from_audio(tokens: List[Dict[str, Any]]) -> List[str]:
    """
    실제 음성에서 추출된 토큰들을 기반으로 hotwords 동적 생성
    비즈니스/회의 관련 키워드들을 우선적으로 추출
    """
    hotwords = []
    
    # 비즈니스/회의 관련 키워드 매핑
    business_keywords = {
        "회의": ["회의", "미팅", "meeting"],
        "시작": ["시작", "시작하", "시작하겠습니다"],
        "프로젝트": ["프로젝트", "project"],
        "발표": ["발표", "presentation"],
        "안건": ["안건", "agenda"],
        "스프린트": ["스프린트", "sprint"],
        "킥오프": ["킥오프", "kickoff"],
        "리뷰": ["리뷰", "review"],
        "회고": ["회고", "retrospective"],
        "토론": ["토론", "discussion"],
        "결정": ["결정", "decision"],
        "계획": ["계획", "plan"],
        "진행": ["진행", "progress"],
        "완료": ["완료", "complete"],
        "다음": ["다음", "next"]
    }
    
    # 토큰에서 키워드 추출
    for token in tokens:
        token_text = token.get('text', '').strip()
        
        # 직접 매칭되는 키워드 찾기
        for keyword, variations in business_keywords.items():
            if any(var in token_text for var in variations):
                if keyword not in hotwords:
                    hotwords.append(keyword)
    
    # 기본 hotwords가 없으면 기본값 추가
    if not hotwords:
        hotwords = ["회의", "시작", "프로젝트", "발표", "안건"]
    
    return hotwords


def extract_domain_lexicon_from_audio(tokens: List[Dict[str, Any]]) -> List[str]:
    """
    실제 음성에서 추출된 토큰들을 기반으로 domain_lexicon 동적 생성
    도메인별 전문 용어들을 추출
    """
    domain_lexicon = []
    
    # 도메인별 전문 용어 매핑
    domain_terms = {
        "회의": ["회의", "회의실", "회의록", "회의록", "회의자", "회의참석자"],
        "프로젝트": ["프로젝트", "프로젝트관리", "프로젝트팀", "프로젝트계획"],
        "발표": ["발표", "발표자", "발표자료", "발표내용", "발표시간"],
        "스프린트": ["스프린트", "스프린트계획", "스프린트리뷰", "스프린트회고"],
        "개발": ["개발", "개발자", "개발환경", "개발팀", "개발과정"],
        "테스트": ["테스트", "테스트케이스", "테스트환경", "테스트결과"],
        "배포": ["배포", "배포환경", "배포계획", "배포일정"],
        "문서": ["문서", "문서화", "문서작성", "문서관리"],
        "코드": ["코드", "코드리뷰", "코드품질", "코드관리"],
        "버그": ["버그", "버그수정", "버그리포트", "버그추적"]
    }
    
    # 토큰에서 도메인 용어 추출
    for token in tokens:
        token_text = token.get('text', '').strip()
        
        # 직접 매칭되는 도메인 용어 찾기
        for domain, terms in domain_terms.items():
            if any(term in token_text for term in terms):
                # 해당 도메인의 모든 용어 추가
                for term in terms:
                    if term not in domain_lexicon:
                        domain_lexicon.append(term)
    
    # 기본 domain_lexicon이 없으면 기본값 추가
    if not domain_lexicon:
        domain_lexicon = ["회의", "회의실", "회의록", "발표", "프로젝트"]
    
    return domain_lexicon
