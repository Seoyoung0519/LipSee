# ONNX Runtime 기반 감정 분류 서비스 (PyTorch 대신)
# - 메모리 사용량 대폭 절약: PyTorch(~800MB) → ONNX Runtime(~50MB)
# - 로컬 onnx/ 폴더의 모델 파일 사용

# ONNX Runtime 기반 감정 분류기 직접 구현
from __future__ import annotations
from typing import Iterable, List, Tuple
from threading import Lock
import os
import json
import numpy as np

from core.config import settings
from schemas.emotions import ECRefinedSegment, ClassifiedSegment

# 감정 라벨 매핑 (원래 모델의 실제 출력)
LABEL_KO_MAP = {
    # 원래 모델의 11가지 감정
    "joy": "기쁨(행복한)",
    "grateful": "고마운",
    "excited": "설레는(기대하는)",
    "loving": "사랑하는",
    "fun": "즐거운(신나는)",
    "daily": "일상적인",
    "thoughtful": "생각이 많은",
    "sadness": "슬픔(우울한)",
    "difficult": "힘듦(지침)",
    "annoyed": "짜증남",
    "worried": "걱정스러운(불안한)",
    
    # 기타 가능한 매핑
    "neutral": "일상적인",
    "anger": "짜증남",
    "anxiety": "걱정스러운(불안한)",
    "fear": "걱정스러운(불안한)",
    "surprise": "설레는(기대하는)",
    "embarrassment": "생각이 많은",
    "confusion": "생각이 많은",
    "hurt": "슬픔(우울한)",
    "etc": "일상적인",
    "others": "일상적인",
}

def _normalize_label(label: str, id2label: dict[int, str]) -> str:
    """모델이 반환하는 라벨을 한국어로 정규화"""
    raw = label
    if raw.isdigit():
        idx = int(raw)
        if idx in id2label:
            raw = id2label[idx]
    if raw.startswith("LABEL_") and raw[6:].isdigit():
        idx = int(raw[6:])
        if idx in id2label:
            raw = id2label[idx]
    low = raw.strip().lower()
    if low in LABEL_KO_MAP:
        return LABEL_KO_MAP[low]
    return raw

# 모델 싱글톤 보장
_onnx_model_singleton = None
_onnx_model_lock = Lock()

class ONNXEmotionClassifier:
    """ONNX Runtime 기반 감정 분류기"""
    
    def __init__(self, onnx_path: str, config_path: str, tokenizer_path: str):
        import onnxruntime as ort
        
        # ONNX 모델 로딩
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        # 모델 설정 로딩
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 토크나이저 로딩
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 모델 입력/출력 정보
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 라벨 매핑
        self.id2label = self.config.get('id2label', {})
        
        print(f"✅ ONNX 모델 로딩 완료: {onnx_path}")
        print(f"📊 모델 입력: {self.input_name}")
        print(f"📊 모델 출력: {self.output_name}")
        print(f"📊 설정 파일: {config_path}")
        print(f"📊 토크나이저: {tokenizer_path}")
        print(f"📊 라벨 매핑: {self.id2label}")
        print(f"📊 모델 입력 형태: {[inp.shape for inp in self.session.get_inputs()]}")
        print(f"📊 모델 출력 형태: {[out.shape for out in self.session.get_outputs()]}")
    
    def tokenize(self, text: str, max_length: int = 128) -> dict:
        """텍스트를 토큰화하여 모델 입력 형태로 변환"""
        encoding = self.tokenizer.encode(text)
        
        # input_ids
        input_ids = encoding.ids[:max_length]
        # 패딩으로 max_length 맞추기
        if len(input_ids) < max_length:
            input_ids += [0] * (max_length - len(input_ids))
        
        # attention_mask (1: 실제 토큰, 0: 패딩)
        attention_mask = [1] * len(encoding.ids[:max_length])
        if len(attention_mask) < max_length:
            attention_mask += [0] * (max_length - len(attention_mask))
        
        # token_type_ids (BERT 스타일, 모두 0)
        token_type_ids = [0] * max_length
        
        return {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64),
            'token_type_ids': np.array([token_type_ids], dtype=np.int64)
        }
    
    def predict_one(self, text: str, max_length: int = 128) -> Tuple[str, float]:
        """단일 텍스트에 대한 감정 예측"""
        if not text or not text.strip():
            return "중립", 0.0
        
        try:
            print(f"🔍 입력 텍스트: {text}")
            tokenized = self.tokenize(text, max_length)
            print(f"🔍 토큰화 결과: input_ids={tokenized['input_ids'].shape}, attention_mask={tokenized['attention_mask'].shape}")
            
            # 모든 입력을 모델에 제공
            model_inputs = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids']
            }
            
            outputs = self.session.run([self.output_name], model_inputs)
            print(f"🔍 모델 출력: {len(outputs)}, 형태: {[o.shape for o in outputs]}")
            
            logits = outputs[0][0]
            print(f"🔍 로짓 형태: {logits.shape}, 값: {logits[:5]}...")
            
            probabilities = self._softmax(logits)
            print(f"🔍 확률 형태: {probabilities.shape}, 값: {probabilities[:5]}...")
            
            predicted_id = np.argmax(probabilities)
            confidence = float(probabilities[predicted_id])
            
            print(f"🔍 예측 ID: {predicted_id}, 신뢰도: {confidence:.3f}")
            
            label = self.id2label.get(str(predicted_id), str(predicted_id))
            print(f"🔍 원본 라벨: {label}")
            
            normalized_label = _normalize_label(label, self.id2label)
            print(f"🔍 정규화된 라벨: {normalized_label}")
            
            return normalized_label, confidence
            
        except Exception as e:
            print(f"⚠️ 예측 오류: {e}")
            import traceback
            traceback.print_exc()
            return "중립", 0.0
    
    def predict(self, texts: Iterable[str], max_length: int = 128) -> List[Tuple[str, float]]:
        """여러 텍스트에 대한 감정 예측"""
        return [self.predict_one(t, max_length) for t in texts]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """소프트맥스 함수"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

def _get_onnx_model() -> ONNXEmotionClassifier:
    """ONNX 모델 싱글톤 인스턴스 반환"""
    global _onnx_model_singleton
    
    if _onnx_model_singleton is None:
        with _onnx_model_lock:
            if _onnx_model_singleton is None:
                # ONNX 모델 경로 설정 (양자화된 모델 우선 사용)
                onnx_dir = os.path.join(os.path.dirname(__file__), '..', 'onnx')
                
                # 양자화된 모델 우선 시도
                int8_path = os.path.join(onnx_dir, 'model.int8.onnx')
                if os.path.exists(int8_path):
                    onnx_path = int8_path
                    print(f"🚀 양자화된 int8 모델 사용: {int8_path}")
                else:
                    onnx_path = os.path.join(onnx_dir, 'model.onnx')  # 원본 모델
                    print(f"📊 기본 fp32 모델 사용: {onnx_path}")
                
                config_path = os.path.join(onnx_dir, 'config.json')
                tokenizer_path = os.path.join(onnx_dir, 'tokenizer.json')
                
                # 파일 존재 확인
                if not all(os.path.exists(p) for p in [onnx_path, config_path, tokenizer_path]):
                    raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {onnx_dir}")
                
                # 모델 크기 정보 출력
                model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"📊 사용할 모델: {os.path.basename(onnx_path)} ({model_size_mb:.1f}MB)")
                
                _onnx_model_singleton = ONNXEmotionClassifier(
                    onnx_path, config_path, tokenizer_path
                )
    
    return _onnx_model_singleton

def classify_segments(
    segs: Iterable[ECRefinedSegment],
    add_prefix: bool = True
) -> List[ClassifiedSegment]:
    """ECRefinedSegment 리스트를 받아 감정 분류 후 ClassifiedSegment로 변환"""
    model = _get_onnx_model()
    preds = model.predict([s.corrected for s in segs])
    
    out: List[ClassifiedSegment] = []
    for s, (emo, score) in zip(segs, preds):
        final_text = f"({emo}) {s.corrected}" if add_prefix else s.corrected
        out.append(ClassifiedSegment(
            id=s.id, start=s.start, end=s.end,
            original=s.original, picked_candidate=s.picked_candidate,
            gain=s.gain, corrected=s.corrected,
            emotion=emo, score=score, final_text=final_text
        ))
    return out
