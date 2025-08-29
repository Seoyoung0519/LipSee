# EC Model FastAPI API

발음 오류나 음성 인식 불확실 구간을 보수적으로 교정하는 한국어 **음성 자막 정교화 API**입니다.  
AI 기반 AV-ASR 자막을 입력받아, 명사 위주의 오류 교정과 띄어쓰기/맞춤법 보정을 수행합니다.  
최종 자막 출력은 KoBART 기반 텍스트 정교화 모델을 사용합니다.

---

## 프로젝트 구조

```bash
LipSee_ECmodel/
├─ app.py                        # FastAPI 서버 진입점
├─ config.py                     # config_ec.json 로더
├─ requirements.txt              # 의존 패키지 목록
│
├─ 📂 routes/
│   └─ correct.py                # /ec/correct endpoint 라우터 정의
│
├── 📂 schema/
│   ├─ asr_payload.py          # 요청 스키마 정의
|   └─ ec.py                   # 전체 EC 파이프라인 처리
│
├─ 📂 service/
│   ├─ kobart_corrector.py       # KoBART 기반 문장 마무리 정리기
│   ├─ phonetic.py               # 자모 기반 음운 거리 계산 + 후보 생성
│   ├─ pos.py                    # 명사 추출을 위한 품사 분석기 (kiwipiepy 사용)
│   ├─ rerank.py                 # 후보 문장 점수화 (MLM + Seq2Seq 기반 gain 계산)
│   ├─ postprocess.py            # 최종 문장 후처리 (띄어쓰기, 중복 문장 제거 등)
│   └─ batching.py               # 다중 문장 배치 처리 유틸
│
└─ 📂 models/
    └─ kobart_ec/                # 학습된 KoBART 정교화 모델 (핵심 파일만)
        ├── config.json          # KoBART 모델 설정 파일 (HuggingFace 형식)
        ├── model.safetensors    # 파인튜닝된 KoBART 모델 가중치
        ├── tokenizer.json       # 토크나이저 설정
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── ec_config.json       # EC pipeline 설정(json, gain 임계값 등)
```

===

## 설치 & 실행 가이드

### 1. 가상환경 생성 및 패키지 설치
```python 
-m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 모델 경로 확인
```bash
# config.py에 정의된 경로 확인:
KOBART_EC_PATH = "./models/kobart_ec"
```
ec_config.json, model.safetensors, tokenizer.json 등의 파일이 위 경로에 존재해야 합니다.

### 3. FastAPI 실행
```bash
uvicorn app:app --reload --port 8000
```

### 4. Swagger 테스트

* Docs: http://localhost:8000/docs

* Health Check: http://localhost:8000/healthz

===

## API 사용법

### 📍 엔드포인트
```bash
POST /ec/correct
```
### 📥 입력 예시 (application/json)
```json
{
  "request_id": "req_20241226_143022_a1b2c3d4",
  "model_version": { "av_asr": "av-asr-0.9.3" },
  "segments": [
    {
      "id": "seg_00000",
      "start": 0.0,
      "end": 2.0,
      "text": "지금부터 회웨을 시작하겠읍다",
      "confidence": 0.893,
      "no_speech_prob": 0.02,
      "frame_entropy": 0.156,
      "words": [
        { "text": "지금부터", "t0": 0.0, "t1": 0.6, "logprob": -0.034, "confidence": 0.93 }
      ],
      "nbest": [
        { "rank": 1, "text": "지금부터 회의를 시작하겠습니다", "score": -2.456, "confidence": 0.893 },
        { "rank": 2, "text": "지금부터 홈페이지를 시작하겠습니다", "score": -4.123, "confidence": 0.156 }
      ]
    }
  ],
  "hotwords": ["회의", "안건", "스프린트"],
  "domain_lexicon": ["회의", "회의록", "프로젝트"]
}
```

### 📤 출력 예시
```json
{
  "request_id": "req_20241226_143022_a1b2c3d4",
  "model_version": { "av_asr": "av-asr-0.9.3" },
  "segments": [
    {
      "id": "seg_00000",
      "start": 0,
      "end": 2,
      "original": "지금부터 회웨을 시작하겠읍다",
      "picked_candidate": "지금부터 회의를 시작하겠습니다",
      "gain": 2.037,
      "corrected": "지금부터 회의를 시작하겠습니다."
    }
  ]
}
```

===

## EC 모델 처리 흐름
```bash
flowchart TD
  A[AV-ASR 자막 JSON 입력]
  B[신뢰도 낮은 명사 후보 추출]
  C[발음 유사 후보 생성]
  D[MLM & Seq2Seq 재랭킹]
  E[최적 후보 선택 + gain 계산]
  F[KoBART 정교화]
  G[postprocess 및 출력]

  A --> B --> C --> D --> E --> F --> G
  ```

===

## FastAPI 처리 흐름
```bash
flowchart TD
  U[사용자 JSON 입력] --> R[POST /ec/correct]
  R --> V[ECRequest 변환]
  V --> P[ec() 파이프라인 실행]
  P --> O[JSON 응답 반환]
```

===

## 테스트 결과 예시 (Swagger)
| 입력              | 출력                |
| --------------- | ----------------- |
| 지금부 회웨을 시작하겠읍다 | 지금부터 회의를 시작하겠습니다. |

===

## 모델 필터링 전략: Guard CER

* CER 기반 보수적 적용 (오답 필터)
* config.py:
```python
GUARD_CER = 0.05
```
* jiwer.cer을 기준으로 오답률이 높으면 원문 유지하여 과교정 방지