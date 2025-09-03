# Enhanced AV-ASR System for EC Model Integration

## 📋 프로젝트 개요

Enhanced AV-ASR (Audio-Visual Automatic Speech Recognition) 시스템은 **Wav2Vec2 + Whisper**의 오디오 융합을 통해 고품질 자막을 생성하고, EC(Error Correction) 모델과의 연동을 위한 상세한 출력을 제공합니다.

### 🎯 주요 특징
- **오디오 융합**: Wav2Vec2 + Whisper
- **앙상블 오디오 융합**: 가중 평균, 최대값, 적응형 융합 지원
- **Enhanced CTC Decoder**: GELU 활성화, Beam Search (beam_size=5)
- **한국어 특화**: 한국어 후처리 및 신뢰도 기반 필터링
- **EC 모델 연동**: 토큰/단어별 상세 정보, n-best 후보, 프레임 엔트로피
- **실시간 처리**: 25fps 프레임 기반 윈도우 처리
- **✨ 25fps 적응형 자동 키워드 생성**: 신뢰도 기반 적응형 임계값, 동적 도메인 패턴, 프레임별 키워드 매핑

## 🏗️ 프로젝트 구조

```
AV_ASR/
├── server/                         # 서버 모듈
│   ├── pipeline/                   # 파이프라인
│   │   └── ec_integration_pipeline.py  # EC 모델 연동 파이프라인
│   ├── models/                     # 모델 클래스들
│   │   ├── config.py              # 설정 관리
│   │   ├── wav2vec2_encoder.py    # Wav2Vec2 인코더
│   │   ├── whisper_encoder.py     # Whisper 인코더
│   │   ├── exceptions.py          # 예외 처리
│   │   └── utils.py               # 유틸리티 함수
│   └── utils/                      # 유틸리티
│       └── srt.py                 # SRT 자막 생성
├── app.py                         # FastAPI 메인 애플리케이션
├── run_server.py                  # 서버 실행 스크립트
├── requirements.txt               # Python 의존성
└── README.md                     # 프로젝트 문서
```

## 📡 API 엔드포인트

### **API 문서**
- **웹 문서**: `https://av-asr.onrender.com/docs` - 간단한 HTML 기반 API 문서
- **헬스 체크**: `https://av-asr.onrender.com/v1/health` - 시스템 상태 확인

### **1. Enhanced AV-ASR 추론** (`POST /v1/enhanced_infer`)

#### **입력 파라미터**
```python
{
    "file": "UploadFile",                    # 영상/오디오 파일
    "format": "json|srt|both",              # 출력 포맷 (기본값: json)
    "language": "ko|en",                    # 언어 (기본값: ko)
    "return_words": bool,                   # 단어 단위 정보 (기본값: true)
    "audio_fusion_method": str,             # 오디오 융합 방식
    "audio_fusion_alpha": float,            # 오디오 융합 가중치 (0.0~1.0)
    "hotwords": str,                        # 핫워드 (쉼표로 구분) - 자동 생성되므로 선택사항
    "domain_lexicon": str                   # 도메인 어휘 (쉼표로 구분) - 자동 생성되므로 선택사항
}
```

#### **✨ 25fps 단위 자동 키워드 생성 기능 (NEW)**
- **자동 Hotwords**: 신뢰도(≥0.6)와 빈도(≥2)를 기반으로 중요 단어 자동 추출
- **자동 Domain Lexicon**: 한국어 비즈니스 패턴 매칭으로 도메인 어휘 생성
- **프레임별 매핑**: 각 프레임(40ms)마다 키워드 시간 정보 제공
- **자동 저장**: 생성된 키워드 정보를 JSON 파일로 자동 저장
- **실시간 분석**: 25fps 비디오 프레임과 완벽 동기화

**새로운 출력 필드:**
```json
{
    "hotwords": ["회의", "시작", "프로젝트", "발표"],  // 자동 생성된 핫워드
    "domain_lexicon": ["회의", "회의실", "발표", "프로젝트", "업무"]  // 자동 생성된 도메인 어휘
}
```

#### **오디오 융합 방식**
- **`weighted`**: 가중 평균 (기본값: Wav2Vec2 60%, Whisper 40%)
- **`max`**: 최대값 선택
- **`adaptive`**: 신뢰도 기반 적응형
- **`concat`**: 연결 융합 (기존 방식)

#### **출력 형식 (EC 모델 연동용)**
```json
{
    "request_id": "req_20241226_143022_a1b2c3d4",
    "model_version": {
        "av_asr": "av-asr-0.9.4",
        "audio_encoder": "wav2vec2-kspon-pt",
        "audio_encoder2": "whisper-encoder-large-v3",
        "visual_encoder": "removed",
        "ctc_decoder": "enhanced_beam_lm"
    },
    "media": {
        "duration_sec": 10.5,
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
            "name": "removed",
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
            "end": 2.0,
            "text": "지금부터 회의를 시작하겠습니다",
            "confidence": 0.893,
            "no_speech_prob": 0.02,
            "frame_entropy": 0.156,
            "tokens": [
                {
                    "text": "지금",
                    "t0": 0.0,
                    "t1": 0.4,
                    "f0": 0,
                    "f1": 10,
                    "logprob": -0.023,
                    "confidence": 0.95
                },
                {
                    "text": "부터",
                    "t0": 0.4,
                    "t1": 0.6,
                    "f0": 10,
                    "f1": 15,
                    "logprob": -0.045,
                    "confidence": 0.92
                }
            ],
            "words": [
                {
                    "text": "지금부터",
                    "t0": 0.0,
                    "t1": 0.6,
                    "logprob": -0.034,
                    "confidence": 0.93
                },
                {
                    "text": "회의를",
                    "t0": 0.6,
                    "t1": 1.2,
                    "logprob": -0.067,
                    "confidence": 0.89
                }
            ],
            "nbest": [
                {
                    "rank": 1,
                    "text": "지금부터 회의를 시작하겠습니다",
                    "score": -2.456,
                    "confidence": 0.893,
                    "tokens": [...]
                },
                {
                    "rank": 2,
                    "text": "지금부터 홈페이지를 시작하겠습니다",
                    "score": -4.123,
                    "confidence": 0.156,
                    "tokens": [...]
                }
            ]
        }
    ],
    "hotwords": ["회의", "안건", "스프린트", "킥오프"],
    "domain_lexicon": ["회의", "회의실", "회의록", "발표", "프로젝트"]
}
```

### **2. 시스템 정보** (`GET /v1/enhanced_info`)

#### **응답**
```json
{
    "system": "Enhanced AV-ASR System for EC Model Integration",
    "version": "0.9.4",
    "models": {
        "wav2vec2": true,
        "whisper": true,
        "enhanced_ctc_decoder": true
    },
    "audio_fusion_methods": [
        "weighted", "max", "adaptive", "concat"
    ],
    "supported_formats": [
        "mp4", "avi", "mov", "mkv", "wav", "m4a"
    ],
    "features": [
        "앙상블 오디오 융합",
        "Enhanced CTC Decoder with GELU",
        "한국어 특화 후처리",
        "신뢰도 기반 필터링",
        "Beam search 최적화",
        "EC 모델 연동용 출력"
    ]
}
```

### **3. 헬스 체크** (`GET /v1/health`)

#### **응답**
```json
{
    "status": "ok",
    "models": {
        "wav2vec2": true,
        "whisper": true,
        "enhanced_ctc_decoder": true
    },
    "device": "cpu",
    "fps": 25,
    "pipeline": "Enhanced AV-ASR: Wav2Vec2 + Whisper → Enhanced CTC + 25fps 적응형 키워드 생성",
    "ec_model_ready": true,
    "swagger_optimized": true,
    "features": [
        "앙상블 오디오 융합",
        "Enhanced CTC Decoder with GELU",
        "한국어 특화 후처리",
        "신뢰도 기반 필터링",
        "Beam search 최적화",
        "EC 모델 연동용 출력",
        "25fps 적응형 자동 키워드 생성",
        "Swagger UI 성능 최적화"
    ]
}
```

## 🚀 Quick Start

### 1. **환경 설정**
```bash
# Python 3.8+ 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# FFmpeg 설치 (시스템에 따라)
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

### 2. **서버 실행**
```bash
# 서버 실행 스크립트 사용 (권장)
python run_server.py

# 또는 직접 실행
python app.py

# 또는 uvicorn으로 직접 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. **API 테스트**

#### **웹 UI (권장)**
```
http://localhost:8000/docs    # HTML API 문서
http://localhost:8000/v1/health  # 헬스 체크
```

#### **명령줄 테스트**
```bash
# 헬스 체크
curl http://localhost:8000/v1/health

# Enhanced AV-ASR 추론
curl -X POST "http://localhost:8000/v1/enhanced_infer" \
  -F "file=@video.mp4" \
  -F "audio_fusion_method=weighted" \
  -F "audio_fusion_alpha=0.6"
```

## ⚙️ 설정 옵션

### **오디오 융합 설정**
```python
# 가중 평균 앙상블 (권장)
audio_fusion_method = "weighted"
audio_fusion_alpha = 0.6  # Wav2Vec2 60%, Whisper 40%

# 최대값 앙상블
audio_fusion_method = "max"

# 적응형 앙상블
audio_fusion_method = "adaptive"
```

### **CTC 디코더 설정**
```python
beam_size = 5                    # Beam search 크기
confidence_threshold = 0.01      # 신뢰도 임계값
```

## 📊 성능 최적화

### **앙상블 융합 성능 비교**
| 융합 방식 | 정확도 | 신뢰도 | 처리 속도 | 메모리 사용량 |
|-----------|--------|--------|-----------|---------------|
| **가중 평균** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **최대값** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **적응형** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **연결** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### **권장 설정**
```python
# 최적 성능 설정
audio_fusion_method = "weighted"
audio_fusion_alpha = 0.6
beam_size = 5
confidence_threshold = 0.01
```

## 🔧 개발 가이드

### **새로운 융합 방식 추가**
```python
# server/pipeline/ec_integration_pipeline.py
def ensemble_inference_custom(wav2vec2_result, whisper_result, **kwargs):
    """커스텀 융합 방식 구현"""
    # 구현 로직
    return fused_result
```

### **새로운 모델 통합**
```python
# server/models/
class CustomEncoder:
    def __init__(self):
        pass
    
    def transcribe(self, audio):
        # 인코딩 로직
        return result
```

## 📝 변경 이력

- **v0.9.4**: 🚀 **FastAPI 서버 구현 및 코드 정리**
  - FastAPI 서버 구현 및 모듈화된 구조로 정리
  - EC 모델 연동용 출력 형식 완전 유지
  - 25fps 적응형 자동 키워드 생성 기능
  - Swagger UI 성능 최적화
  - 불필요한 파일 정리 및 구조 개선
- **v0.9.3**: ✨ **25fps 적응형 자동 키워드 생성 기능 추가**
  - 적응형 임계값 계산 (대화 품질 기반 자동 조정)
  - 동적 도메인 패턴 생성 (실제 대화 내용 기반)
  - 프레임별 키워드 매핑 및 자동 집계
- **v0.9.0**: 다중 모달 융합 파이프라인 구현
- **v0.8.0**: Wav2Vec2 + Whisper 인코더 분리 구현
- **v0.7.0**: EC 모델 연동용 출력 형식 설계


