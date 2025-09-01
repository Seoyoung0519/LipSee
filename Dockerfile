FROM python:3.9-slim

WORKDIR /app

# 1) 시스템 패키지: execstack 포함 (플래그 해제 도구)
RUN apt-get update && apt-get install -y \
    gcc g++ \
    execstack \
    && rm -rf /var/lib/apt/lists/*

# (참고) 컨테이너 내 sysctl은 효과 없으니 제거
# RUN echo 'kernel.yama.ptrace_scope = 0' >> /etc/sysctl.conf

# 2) Python 의존성 설치
COPY requirements.txt .
# 혹시 겹침 방지: onnxruntime / onnxruntime-cpu 동시 설치 금지
RUN pip uninstall -y onnxruntime onnxruntime-cpu || true
# 원하는 쪽 하나만: 예) onnxruntime-cpu==1.16.3
# (이미 requirements.txt에 박아놨다면 아래 줄은 주석 처리 가능)
# RUN pip install --no-cache-dir onnxruntime-cpu==1.16.3
RUN pip install --no-cache-dir -r requirements.txt

# 3) ORT .so 에서 executable stack 플래그 제거 (import 없이 경로 탐색)
#    BEFORE / AFTER 상태를 로그로 출력해 확인 가능
RUN bash -lc '\
  set -euo pipefail; \
  SITE_PKGS=$(python -c "import site,sys; print((site.getsitepackages() or [sys.prefix])[0])"); \
  ORT_CAPIDIR=\"$SITE_PKGS/onnxruntime/capi\"; \
  echo \"site-packages: $SITE_PKGS\"; \
  echo \"onnxruntime/capi: $ORT_CAPIDIR\"; \
  if [ -d \"$ORT_CAPIDIR\" ]; then \
    echo \"[BEFORE] execstack flags:\"; (execstack -q \"$ORT_CAPIDIR\"/*.so* || true); \
    find \"$ORT_CAPIDIR\" -maxdepth 1 -type f -name \"*.so*\" -exec execstack -c {} \; || true; \
    echo \"[AFTER] execstack flags:\"; (execstack -q \"$ORT_CAPIDIR\"/*.so* || true); \
  else \
    echo \"WARN: $ORT_CAPIDIR not found; skip execstack patch\"; \
  fi \
'

# 4) 앱 코드 복사
COPY . .

# 5) 환경변수 (기본값 설정) — 필요시 대시보드에서 override 가능
ENV ONNX_ZIP_URL="https://github.com/Seoyoung0519/LipSee/releases/download/v1.0.0"
ENV ONNXRUNTIME_DISABLE_GPU=1
ENV OMP_NUM_THREADS=1
ENV ORT_NUM_THREADS=1
# Render가 주는 PORT를 우선 사용, 없으면 10000
ENV PORT=10000

EXPOSE 10000

# 6) 단일 CMD만 사용 (중복 금지)
CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
