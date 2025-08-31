# 📌LipSee Emotion Classification API

EC(정교화) 모델의 **출력 자막(corrected)**을 입력받아, 세그먼트 단위로 11가지 감정을 분류합니다.
출력은 JSON 형태와 SRT 파일 형태를 모두 지원합니다.

---

## 🚀 **설치 및 실행 가이드**

### **1. 패키지 설치**
```bash
pip install -r requirements.txt
```

### **2. 서버 실행**
```bash
uvicorn app:app --reload --port 8081
```

### **3. Swagger UI 접속**
- **API 문서**: http://127.0.0.1:8081/docs
- **ReDoc 문서**: http://127.0.0.1:8081/redoc
- **헬스체크**: http://127.0.0.1:8081/health

---

## 🛠 API 엔드포인트
- POST /classify
    - EC 모델 출력 구조로 감정 라벨링

- POST /srt
  - 감정이 포함된 SRT 자막을 생성

- GET /health
  - 서버 상태 확인용 헬스 체크

---

## 📚 API 설명

### 1. POST /classify

- 설명: EC 모델 출력 구조(segments[].corrected)로 감정 분류
- 입력: ECRefinedSegment 리스트 (start/end 사용, picked_candidate, gain 포함)
- 출력: 각 세그먼트별 emotion, score, final_text

### 2. POST /srt

- 설명: 감정 분류 후 SRT 파일 생성 또는 JSON 응답
- 입력: SrtRequest + format 파라미터
- format 파라미터:
  - `file` (기본값): SRT 파일 다운로드
  - `json`: JSON 응답 (SRT 텍스트 포함)
- 출력: SRT 파일 또는 JSON 응답

---

## 💻 클라이언트 예시

### Python requests
```python
import requests

url = "http://localhost:8081/classify"
payload = {
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

res = requests.post(url, json=payload)
print(res.json())
```

### curl
```bash
curl -X POST "http://localhost:8081/classify" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

## 📥 요청/응답 예시

### 요청 예시 (POST `/classify`)
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

**참고**: `final_prefix_emotion` 필드는 선택적입니다. 입력하지 않으면 기본값 `true`가 적용됩니다.

### 응답 예시
```json
{
  "request_id": "req_003",
  "model_version": {
    "av_asr": "av-asr-0.9.3"
  },
  "segments": [
    {
      "id": "seg_00004",
      "start": 8,
      "end": 10,
      "original": "아직 문제가 많이 남아있어",
      "picked_candidate": "아직 문제가 많이 남아있어요.",
      "gain": 1.1,
      "corrected": "아직 문제가 많이 남아있어요.",
      "emotion": "걱정스러운(불안한)",
      "score": 0.7187399864196777,
      "final_text": "(걱정스러운(불안한)) 아직 문제가 많이 남아있어요."
    },
    {
      "id": "seg_00005",
      "start": 10,
      "end": 12,
      "original": "정말 힘들고 어려워",
      "picked_candidate": "정말 힘들고 어려워요.",
      "gain": 0.9,
      "corrected": "정말 힘들고 어려워요.",
      "emotion": "슬픔(우울한)",
      "score": 0.9921295046806335,
      "final_text": "(슬픔(우울한)) 정말 힘들고 어려워요."
    }
  ],
  "num_segments": 2
}
```

---

### 요청 예시 (POST `/srt`)
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

**참고**: `final_prefix_emotion`과 `filename` 필드는 선택적입니다. 입력하지 않으면 기본값이 적용됩니다.

### 응답 예시 (JSON 응답 - format=json)
```json
{
  "request_id": "req_20241226_143022_a1b2c3d4",
  "model_version": {
    "av_asr": "av-asr-0.9.3"
  },
  "segments": [
    {
      "id": "seg_00000",
      "start": 0,
      "end": 2,
      "original": "지금부터 회웨을 시작하겠읍다",
      "picked_candidate": "지금부터 회의를 시작하겠습니다",
      "gain": 2.037,
      "corrected": "지금부터 회의를 시작하겠습니다.",
      "emotion": "설레는(기대하는)",
      "score": 0.8030101656913757,
      "final_text": "(설레는(기대하는)) 지금부터 회의를 시작하겠습니다."
    }
  ],
  "srt_text": "1\n00:00:00,000 --> 00:00:02,000\n(설레는(기대하는)) 지금부터 회의를 시작하겠습니다.\n",
  "filename": "meeting_emotion.srt"
}
```

### 응답 예시 (SRT 파일 다운로드 - format=file)
```
1
00:00:00,000 --> 00:00:02,000
(설레는(기대하는)) 지금부터 회의를 시작하겠습니다.
```

---

## ⚠️ 유의사항
- 입력 세그먼트는 EC 모델 출력 구조(original, corrected, picked_candidate, gain 등)를 그대로 사용합니다.

- 감정 분류에는 corrected 필드만 사용합니다.

- 모델(`nlp04/korean_sentiment_analysis_kcelectra`)은 11가지 감정을 분류합니다. (분노, 불안, 혐오, 두려움, 슬픔, 기쁨, 놀람, 중립, 당황, 상처, 기타)

- 긴 문장은 토크나이저 `max_length`(기본 256 토큰) 기준으로 잘립니다.

- 응답의 `score`는 softmax 확신도이므로, 후처리에서 임계치를 적용할 수 있습니다.

- `/srt` 응답은 MIME 타입 `application/x-subrip`이며, `Content-Disposition` 헤더로 파일 다운로드가 가능합니다.

- 시간 정보는 초 단위(`start`, `end`)로 입력받으며, SRT 생성 시 밀리초로 자동 변환됩니다.
