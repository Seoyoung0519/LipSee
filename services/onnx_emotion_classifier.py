# ONNX Runtime 기반 감정 분류 서비스
# - PyTorch 대신 ONNX Runtime 사용으로 메모리 절약
# - 로컬 onnx/ 폴더의 모델 파일 사용

from __future__ import annotations
from typing import Iterable, List, Tuple
from threading import Lock
import os
import json
import numpy as np

from core.config import settings
from schemas.emotions import ECRefinedSegment, ClassifiedSegment

# 감정 라벨 매핑 (모델 출력에 맞게 조정 필요)
LABEL_KO_MAP = {
    "anger": "분노",
    "anxiety": "불안", 
    "disgust": "혐오",
    "fear": "두려움",
    "sadness": "슬픔",
    "joy": "기쁨",
    "surprise": "놀람",
    "neutral": "중립",
    "embarrassment": "당황",
    "confusion": "당황",
    "hurt": "상처",
    "etc": "기타",
    "others": "기타",
}

def _normalize_label(label: str, id2label: dict[int, str]) -> str:
    """모델이 반환하는 라벨을 한국어로 정규화"""
    raw = label
    # 숫자 라벨 처리
    if raw.isdigit():
        idx = int(raw)
        if idx in id2label:
            raw = id2label[idx]
    # LABEL_3 같은 포맷 처리
    if raw.startswith("LABEL_") and raw[6:].isdigit():
        idx = int(raw[6:])
        if idx in id2label:
            raw = id2label[idx]
    # 영문 라벨 매핑
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
        
        # ONNX 모델 로딩 (양자화된 모델 최적화)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        
        # 양자화된 모델인지 확인
        is_int8 = 'int8' in onnx_path.lower()
        if is_int8:
            print("🔧 int8 모델 최적화 설정 적용")
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']  # CPU만 사용
        )
        
        # 모델 설정 로딩
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 토크나이저 로딩
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 모델 입력/출력 정보
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        
        # 라벨 매핑
        self.id2label = self.config.get('id2label', {})
        
        # 모델 타입 확인 및 출력
        model_type = "int8 (양자화)" if 'int8' in onnx_path.lower() else "fp32 (기본)"
        print(f"✅ ONNX 모델 로딩 완료: {onnx_path}")
        print(f"🔍 모델 타입: {model_type}")
        print(f"📊 모델 입력: {self.input_names}")
        print(f"📊 모델 출력: {self.output_name}")
        
        # 양자화된 모델 사용 시 추가 정보
        if 'int8' in onnx_path.lower():
            print("💡 int8 모델: 메모리 사용량 감소, 추론 속도 향상")
    
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
            # 토큰화
            tokenized = self.tokenize(text, max_length)
            
            # 모든 입력을 모델에 제공
            model_inputs = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids']
            }
            
            # ONNX 추론
            outputs = self.session.run(
                [self.output_name], 
                model_inputs
            )
            
            # 결과 처리
            logits = outputs[0][0]  # 첫 번째 배치, 첫 번째 시퀀스
            probabilities = self._softmax(logits)
            
            # 최고 확률의 라벨 선택
            predicted_id = np.argmax(probabilities)
            confidence = float(probabilities[predicted_id])
            
            # 라벨 정규화
            label = self.id2label.get(str(predicted_id), str(predicted_id))
            normalized_label = _normalize_label(label, self.id2label)
            
            return normalized_label, confidence
            
        except Exception as e:
            print(f"⚠️ 예측 오류: {e}")
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
                # ONNX 모델 경로 설정 (환경변수 우선, 없으면 상대 경로)
                onnx_dir = os.getenv('ONNX_MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', 'onnx'))
                print(f"🔍 ONNX 모델 경로: {onnx_dir}")
                print(f"🔍 환경변수 ONNX_MODEL_PATH: {os.getenv('ONNX_MODEL_PATH')}")
                int8_path = os.path.join(onnx_dir, 'model.int8.onnx')
                fp32_path = os.path.join(onnx_dir, 'model.onnx')
                
                # 양자화된 모델이 있으면 사용, 없으면 기본 모델 사용
                if os.path.exists(int8_path):
                    onnx_path = int8_path
                    print(f"🚀 양자화된 int8 모델 사용: {int8_path}")
                else:
                    onnx_path = fp32_path
                    print(f"📊 기본 fp32 모델 사용: {fp32_path}")
                
                config_path = os.path.join(onnx_dir, 'config.json')
                tokenizer_path = os.path.join(onnx_dir, 'tokenizer.json')
                
                # 파일 존재 확인
                if not all(os.path.exists(p) for p in [onnx_path, config_path, tokenizer_path]):
                    raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {onnx_dir}")
                
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
            id=s.id,
            start=s.start,
            end=s.end,
            original=s.original,
            picked_candidate=s.picked_candidate,
            gain=s.gain,
            corrected=s.corrected,
            emotion=emo,
            score=score,
            final_text=final_text
        ))
    return out
