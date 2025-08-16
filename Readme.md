# 📌 EC (KoBART 정교화) API — v2

---

AV-ASR 모델이 출력한 자막(segments)을 입력으로 받아, KoBART 기반 맞춤법·띄어쓰기 교정을 수행한 결과를 반환하는 FastAPI 서비스입니다.
출력에서는 id / start_ms / end_ms를 반드시 보존하며, confidence, words 같은 메타데이터는 들어오면 그대로 돌려줍니다.

---

## 🚀 설치 & 실행

1. 가상환경 생성 및 의존성 설치
~~~
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
~~~
2. KoBART 파인튜닝 체크포인트 경로 지정
~~~
# Hugging Face repo 이름 또는 로컬 디렉토리
export KOBART_EC_PATH=/path/to/your/kobart-ec-ckpt
~~~
3. 서버 실행
~~~
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
~~~

---

## 🛠 API 엔드포인트

### Health Check

- GET /v2/health
    + 서버 및 모델 상태 확인

### 예시
~~~
curl -s http://localhost:8000/v2/health
~~~

### 응답
~~~
{
  "status": "ok",
  "device": "cpu",
  "model_path": "/path/to/your/kobart-ec-ckpt"
}
~~~

---

### 교정 API
- POST /v2/correct
    + AV-ASR segments 배열을 입력받아 교정된 자막을 반환합니다.

### 요청 예시
~~~
curl -X POST "http://localhost:8000/v2/correct" \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {
        "id": "seg_00001",
        "start_ms": 0,
        "end_ms": 1200,
        "text": "안녕하세요 모두",
        "confidence": 0.90,
        "words": [
          {"start_ms": 0, "end_ms": 350, "text": "안녕하세요", "confidence": 0.88},
          {"start_ms": 400, "end_ms": 1200, "text": "모두", "confidence": 0.92}
        ]
      },
      {
        "id": "seg_00002",
        "start_ms": 1200,
        "end_ms": 2500,
        "text": "회의를 시작합니다",
        "confidence": 0.87
      }
    ]
  }'
~~~
### 응답 예시
~~~
{
  "segments": [
    {
      "id": "seg_00001",
      "start_ms": 0,
      "end_ms": 1200,
      "original": "안녕하세요 모두",
      "corrected": "안녕하세요, 모두.",
      "confidence": 0.9,
      "words": [
        {"start_ms": 0, "end_ms": 350, "text": "안녕하세요", "confidence": 0.88},
        {"start_ms": 400, "end_ms": 1200, "text": "모두", "confidence": 0.92}
      ]
    },
    {
      "id": "seg_00002",
      "start_ms": 1200,
      "end_ms": 2500,
      "original": "회의를 시작합니다",
      "corrected": "회의를 시작합니다.",
      "confidence": 0.87,
      "words": null
    }
  ]
}
~~~

---

## 🧩 클라이언트 예시

### Python (requests)
~~~
import requests

payload = {
    "segments": [
        {"id":"seg_00001","start_ms":0,"end_ms":1200,"text":"안녕하세요 모두","confidence":0.9},
        {"id":"seg_00002","start_ms":1200,"end_ms":2500,"text":"회의를 시작합니다","confidence":0.87}
    ]
}

resp = requests.post("http://localhost:8000/v2/correct", json=payload)
print(resp.json())
~~~

### Node.js (fetch)
~~~
const payload = {
  segments: [
    { id: "seg_00001", start_ms: 0, end_ms: 1200, text: "안녕하세요 모두", confidence: 0.9 },
    { id: "seg_00002", start_ms: 1200, end_ms: 2500, text: "회의를 시작합니다", confidence: 0.87 }
  ]
};

fetch("http://localhost:8000/v2/correct", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
})
  .then(r => r.json())
  .then(console.log)
  .catch(console.error);
~~~

---

⚠️ 유의사항

- id / start_ms / end_ms는 절대 변경하지 않습니다.
→ 자막 싱크 안정성 보장

- confidence / words는 옵션 필드
→ 들어오면 응답에 그대로 에코, 안 들어오면 null/생략

- KoBART가 문장을 합치거나 나눠 세그먼트 수가 달라지면 원문 유지
→ 시간축 깨짐 방지

- KoBART 모델은 교정용 파인튜닝된 체크포인트를 사용해야 합니다
(요약용 KoBART 그대로 쓰면 과도 재서술 위험)