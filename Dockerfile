# execstack 설치 가능한 안정 베이스
FROM python:3.9-bullseye
# 또는: FROM python:3.9-bookworm

WORKDIR /app

# execstack 포함
RUN apt-get update && apt-get install -y \
    gcc g++ execstack \
    && rm -rf /var/lib/apt/lists/*

# 의존성
COPY requirements.txt .
# 혹시 중복 설치 방지
RUN pip uninstall -y onnxruntime onnxruntime-cpu || true
# 원하는 한 가지로만 설치: requirements.txt에 이미 명시되어 있으면 아래 줄은 생략 가능
# RUN pip install --no-cache-dir onnxruntime-cpu==1.16.3
RUN pip install --no-cache-dir -r requirements.txt

# onnxruntime .so에서 execstack 플래그 제거 (import 없이 경로 탐색)
RUN bash -lc '\
  set -euo pipefail; \
  SITE_PKGS=$(python -c "import site,sys; print((site.getsitepackages() or [sys.prefix])[0])"); \
  ORT_CAPIDIR="$SITE_PKGS/onnxruntime/capi"; \
  echo "site-packages: $SITE_PKGS"; \
  echo "onnxruntime/capi: $ORT_CAPIDIR"; \
  if [ -d "$ORT_CAPIDIR" ]; then \
    echo "[BEFORE]"; (execstack -q "$ORT_CAPIDIR"/*.so* || true); \
    find "$ORT_CAPIDIR" -maxdepth 1 -type f -name "*.so*" -exec execstack -c {} \; || true; \
    echo "[AFTER]";  (execstack -q "$ORT_CAPIDIR"/*.so* || true); \
  else \
    echo "WARN: $ORT_CAPIDIR not found"; \
  fi \
'

# 앱 복사
COPY . .

# 기본 ENV
ENV ONNX_ZIP_URL="https://github.com/Seoyoung0519/LipSee/releases/download/v1.0.0"
ENV ONNXRUNTIME_DISABLE_GPU=1
ENV OMP_NUM_THREADS=1
ENV ORT_NUM_THREADS=1
ENV PORT=10000

EXPOSE 10000

# 단일 CMD만!
CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
