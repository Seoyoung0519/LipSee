# LipSee Emotion Classification API
# ONNX Runtime 기반 감정 분류 서비스

# 안정적인 베이스 이미지 사용 (execstack 대신 다른 방법 사용)
FROM python:3.9-slim-bullseye

# 메타데이터
LABEL maintainer="LipSee Team"
LABEL version="2.0.0"
LABEL description="FastAPI 기반 감정 분류 API with ONNX Runtime"

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (execstack 제거, 필수 패키지만)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 의존성 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ONNX Runtime 실행 파일 스택 문제 해결 (execstack 없이)
RUN python -c "import site; import os; import sys; print('🔧 ONNX Runtime 실행 파일 스택 문제 해결 중...'); site_packages = site.getsitepackages()[0]; ort_capi_dir = os.path.join(site_packages, 'onnxruntime', 'capi'); print(f'📁 ONNX Runtime CAPI 디렉토리: {ort_capi_dir}') if os.path.exists(ort_capi_dir) else print(f'⚠️ ONNX Runtime CAPI 디렉토리를 찾을 수 없음: {ort_capi_dir}')"

# ONNX Runtime 테스트 및 설정
RUN python -c "import onnxruntime as ort; print(f'✅ ONNX Runtime 로딩 성공: {ort.__version__}'); session_options = ort.SessionOptions(); session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; session_options.intra_op_num_threads = 1; session_options.inter_op_num_threads = 1; print('✅ ONNX Runtime 설정 완료')"

# 애플리케이션 코드 복사
COPY . .

# 보안을 위한 비root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# 환경 변수 설정 (ONNX Runtime 문제 해결을 위한 추가 설정)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ONNX_ZIP_URL="https://github.com/Seoyoung0519/LipSee/releases/download/v1.0.0"
ENV ONNXRUNTIME_DISABLE_GPU=1
ENV ONNXRUNTIME_PROVIDER=CPUExecutionProvider
ENV OMP_NUM_THREADS=1
ENV ORT_NUM_THREADS=1
ENV PORT=10000
# 실행 파일 스택 문제 해결을 위한 추가 환경 변수
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/onnxruntime/capi
ENV PYTHONPATH=/app:/usr/local/lib/python3.9/site-packages

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 포트 노출
EXPOSE 10000

# 애플리케이션 실행 (환경 변수 사용)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
