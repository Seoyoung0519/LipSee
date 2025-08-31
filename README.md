# 📖 LipSee Conductor (Orchestrator)

LipSee Conductor는 화상회의 영상 → **자막 생성(AV-ASR) → 정교화(EC) → 감정 라벨링 및 SRT 생성**  
까지 **원클릭 파이프라인**을 제공하는 백엔드 서비스입니다.  

---

## 📂 프로젝트 구조
```bash
Lipsee-orchestrator/
├─ app/
│  ├─ main.py          # FastAPI 엔트리포인트 (오케스트레이터 API)
│  ├─ clients.py       # Render에 배포된 AV-ASR, EC, Emotion 서버 호출 래퍼
│  ├─ utils.py         # config 로딩, warmup, retry, fetch 유틸
│  ├─ schemas.py       # 응답 스키마 정의 (JSON 디버그용)
│
├─ config/
│  └─ servers.json     # 각 서버 URL 및 웜업 엔드포인트 설정
│
├─ requirements.txt    # Python 의존성
└─ README.md
```

---

## ⚙️ 서버 동작 흐름

### 1. AV-ASR (`/v1/enhanced_infer`)  
- 입력: 회의 영상(mp4 등)  
- 출력: 1차 자막(JSON, n-best 후보 포함)

### 2. EC (`/ec/correct`)  
- 입력: AV-ASR JSON  
- 출력: 정교화된 자막(JSON, `segments[].corrected`, `gain` 포함)

### 3. Emotion (`/classify` → `/srt`)  
- 입력: EC JSON (`segments[].corrected`)  
- `/classify`: 감정 라벨과 점수, 최종 문장(`final_text`) 생성  
- `/srt`: 감정 라벨이 포함된 최종 SRT 파일 생성  

---

## 🔄 진행 플로우

```bash
flowchart TD
    A[회의 영상 업로드/URL] --> B[AV-ASR 서버]
    B --> C[EC 서버]
    C --> D[Emotion 서버 /classify]
    D --> E[Emotion 서버 /srt]
    E --> F[최종 SRT 파일 출력]
```

---

## 📡 API 엔드포인트 (Orchestrator)

**1. 헬스체크**
```http
GET /v1/health
```

**2. 서버 웜업 (콜드스타트 방지)**
```http
POST /v1/warmup
```

**3. 메인 파이프라인 실행**
```http
POST /v1/pipeline/process
```

**요청 방식**

- file: 로컬 업로드 파일(mp4, wav 등)

- video_url: 외부 URL (둘 중 하나 필수)

- 추가 옵션: language, diarize, final_prefix_emotion 등

**응답**

- 항상 SRT 파일 스트림

- Content-Type: application/x-subrip

- 다운로드 파일명: meeting_emotion.srt

---

## 📝 출력 구조 (예시)

**최종 SRT 파일**
```lua
1
00:00:12,000 --> 00:00:14,000
(일상적인) 다음 주 월요일에 마감일입니다.

2
00:00:14,000 --> 00:00:16,000
(일상적인) 예산은 총 500만원입니다.
```

- 번호는 순차 증가

- 시간 포맷: HH:MM:SS,mmm (쉼표 구분)

- 감정 라벨은 final_prefix_emotion=true 옵션 시 문장 앞에 붙음

---

## 🛠 config/servers.json
```json
{
  "av_asr": {
    "base_url": "https://<your-av-asr>.onrender.com",
    "warmup": [{ "method": "GET", "path": "/v1/enhanced_info" }]
  },
  "ec": {
    "base_url": "https://<your-ec>.onrender.com",
    "warmup": [
      { "method": "GET", "path": "/v2/health" },
      { "method": "GET", "path": "/healthz" },
      { "method": "GET", "path": "/" }
    ]
  },
  "emotion": {
    "base_url": "https://<your-emotion>.onrender.com",
    "warmup": [{ "method": "GET", "path": "/health" }],
    "defaults": {
      "final_prefix_emotion": true,
      "srt_format": "file",
      "srt_filename": "meeting_emotion.srt"
    }
  },
  "http": {
    "timeout_seconds": 180,
    "retry_max": 3,
    "retry_backoff_seconds": 2.0,
    "download_limit_mb": 800
  }
}
```

---

## 🚀 실행 방법
```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 💡 활용 포인트

- Render에 배포된 **세 개의 모델 서버(AV-ASR, EC, Emotion)**를 자동으로 깨워 호출 → 최종 자막까지 한 번에 생성

- 콜드스타트 대응: 서버 기동 시/요청 직전 warmup 호출

- 재시도 & 백오프: 일시적 오류에 대한 자동 복구

- 최종 산출물은 SRT 파일: 바로 회의 기록, 영상 자막으로 활용 가능