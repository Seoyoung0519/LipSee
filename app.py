#!/usr/bin/env python3
"""
Enhanced AV-ASR FastAPI Server for EC Model Integration

Enhanced AV-ASR 시스템의 FastAPI 메인 애플리케이션
Wav2Vec2 + Whisper + Enhanced CTC Decoder + 25fps 적응형 키워드 생성
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Literal
import uvicorn

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'server'))

# Render 환경에서 경로 확인
print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path}")
print(f"Server directory exists: {os.path.exists(os.path.join(current_dir, 'server'))}")
print(f"Models directory exists: {os.path.exists(os.path.join(current_dir, 'server', 'models'))}")

# 디렉토리 구조 확인
if os.path.exists(os.path.join(current_dir, 'server')):
    server_contents = os.listdir(os.path.join(current_dir, 'server'))
    print(f"Server directory contents: {server_contents}")
    
    if 'models' in server_contents:
        models_contents = os.listdir(os.path.join(current_dir, 'server', 'models'))
        print(f"Models directory contents: {models_contents}")
    else:
        print("Models directory not found in server folder")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 전역 변수
models_loaded = False
pipeline_module = None


def load_models_lazy():
    """지연 로딩으로 모델 로드 - 안전한 버전"""
    global models_loaded, pipeline_module
    
    if models_loaded and pipeline_module:
        return pipeline_module
    
    try:
        logger.info("🔄 Loading models (Lazy Loading - Safe Mode)...")
        
        # 메모리 사용량 모니터링
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"📊 Memory before loading: {memory_before:.1f} MB")
        
        # 모델 로드
        try:
            from server.pipeline.ec_integration_pipeline import infer_media_for_ec
            pipeline_module = infer_media_for_ec
        except ImportError:
            # 대안 import 방법
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "ec_integration_pipeline", 
                os.path.join(current_dir, "server", "pipeline", "ec_integration_pipeline.py")
            )
            ec_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ec_module)
            pipeline_module = ec_module.infer_media_for_ec
        
        # 메모리 사용량 확인
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        logger.info(f"📊 Memory after loading: {memory_after:.1f} MB (+{memory_increase:.1f} MB)")
        
        # 메모리 제한 확인 (400MB 이하로 제한)
        if memory_after > 400:
            logger.warning(f"⚠️ High memory usage: {memory_after:.1f} MB")
        
        models_loaded = True
        logger.info("✅ Models loaded successfully (Lazy Loading - Safe Mode)")
        return pipeline_module
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        models_loaded = False
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리 - 지연 로딩"""
    global models_loaded
    
    # 시작 시 모델 로드하지 않음 (지연 로딩)
    logger.info("🚀 Starting Enhanced AV-ASR Server (Lazy Loading Mode)...")
    models_loaded = False
    
    yield
    
    # 종료 시 정리
    logger.info("🛑 Shutting down Enhanced AV-ASR Server...")
    if pipeline_module:
        del pipeline_module


# FastAPI 앱 생성
app = FastAPI(
    title="Enhanced AV-ASR System",
    description="Enhanced AV-ASR (Audio-Visual Automatic Speech Recognition) 시스템",
    version="0.9.4",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Enhanced AV-ASR System",
        "version": "0.9.4",
        "status": "running",
        "docs": "/docs",
        "health": "/v1/health"
    }





@app.get("/v1/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        global models_loaded
        
        health_info = {
            "status": "ok" if models_loaded else "degraded",
            "models_loaded": models_loaded,
            "version": "0.9.4"
        }
        
        status_code = 200 if models_loaded else 503
        return JSONResponse(content=health_info, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/v1/enhanced_info")
async def get_enhanced_info():
    """Enhanced AV-ASR 시스템 정보를 반환합니다."""
    
    try:
        info = {
            "system": "Enhanced AV-ASR System",
            "version": "0.9.4",
            "supported_formats": ["mp4", "avi", "mov", "mkv", "wav", "m4a"],
            "supported_languages": ["ko", "en"],
            "output_formats": ["json", "srt", "both"]
        }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        error_msg = f"시스템 정보 조회 실패: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.post(
    "/v1/enhanced_infer",
    summary="Enhanced AV-ASR 자막 생성",
    description="Enhanced AV-ASR 파이프라인을 사용하여 자막을 생성합니다.",
    responses={
        200: {"description": "성공"},
        400: {"description": "잘못된 요청"},
        500: {"description": "서버 내부 오류"}
    }
)
async def enhanced_infer(
    file: UploadFile = File(..., description="영상/오디오 파일 (mp4, avi, mov, mkv, wav, m4a)"),
    format: Literal["json", "srt", "both"] = Form("json", description="출력 포맷"),
    language: Literal["ko", "en"] = Form("ko", description="언어"),
    return_words: bool = Form(True, description="단어 단위 정보"),
    audio_fusion_method: Literal["weighted", "max", "adaptive", "concat"] = Form("weighted", description="오디오 융합 방식"),
    audio_fusion_alpha: float = Form(0.6, description="오디오 융합 가중치 (Wav2Vec2 비율, 0.0~1.0)"),
    hotwords: Optional[str] = Form(None, description="핫워드 (쉼표로 구분) - 자동 생성되므로 선택사항"),
    domain_lexicon: Optional[str] = Form(None, description="도메인 어휘 (쉼표로 구분) - 자동 생성되므로 선택사항")
):
    """
    Enhanced AV-ASR 파이프라인을 사용하여 추론을 수행합니다.
    
    **25fps 단위 자동 키워드 생성 기능:**
    - hotwords와 domain_lexicon이 자동으로 생성됩니다
    - 각 프레임(40ms)마다 키워드 정보를 제공합니다
    - 신뢰도와 빈도를 기반으로 중요한 키워드를 선별합니다
    - 한국어 비즈니스 도메인에 특화된 패턴 매칭을 사용합니다
    
    Args:
        file: 업로드할 미디어 파일
        format: 출력 포맷 ("json", "srt", "both")
        language: 언어 설정
        return_words: 단어 단위 정보 반환 여부
        audio_fusion_method: 오디오 융합 방식
        audio_fusion_alpha: 오디오 융합 가중치
        hotwords: 핫워드 목록 (쉼표로 구분) - 자동 생성되므로 선택사항
        domain_lexicon: 도메인 어휘 목록 (쉼표로 구분) - 자동 생성되므로 선택사항
    
    Returns:
        선택된 포맷에 따른 출력 (25fps 키워드 정보 포함)
    """
    
    # 필요한 import 추가
    from fastapi.responses import FileResponse
    import tempfile
    import json
    import zipfile
    
    # 1) 파일 검증
    try:
        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in [".mp4", ".avi", ".mov", ".mkv", ".wav", ".m4a"]:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다: {suffix}. 지원 형식: mp4, avi, mov, mkv, wav, m4a"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 검증 실패: {e}")
    
    # 2) 파일 저장
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            src_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")
    
    # 3) 매개변수 전처리
    try:
        # 핫워드 파싱
        hotwords_list = None
        if hotwords:
            hotwords_list = [word.strip() for word in hotwords.split(",") if word.strip()]
        
        # 도메인 어휘 파싱
        domain_lexicon_list = None
        if domain_lexicon:
            domain_lexicon_list = [word.strip() for word in domain_lexicon.split(",") if word.strip()]
        
        # 매개변수 검증
        if not (0.0 <= audio_fusion_alpha <= 1.0):
            raise ValueError("audio_fusion_alpha는 0.0과 1.0 사이의 값이어야 합니다")
        
    except ValueError as e:
        # 임시 파일 정리
        try:
            os.unlink(src_path)
        except:
            pass
        raise HTTPException(status_code=400, detail=f"매개변수 검증 실패: {e}")
    
    # 4) Enhanced AV-ASR 추론 (지연 로딩 사용)
    try:
        # 지연 로딩으로 모델 로드
        infer_media_for_ec = load_models_lazy()
        if not infer_media_for_ec:
            raise HTTPException(status_code=503, detail="모델 로드 실패")
        
        result = infer_media_for_ec(
            media_path_or_url=src_path,
            lang=language,
            audio_fusion_method=audio_fusion_method,
            audio_fusion_alpha=audio_fusion_alpha,
            return_words=return_words,
            hotwords=hotwords_list,
            domain_lexicon=domain_lexicon_list
        )
        
        # 5) 임시 파일 정리
        try:
            os.unlink(src_path)
        except:
            pass
        
        # 6) 출력 포맷 처리
        if format == "json":
            return JSONResponse(
                content=result,
                status_code=200,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "X-EC-Model-Ready": "true"
                }
            )
        
        # SRT 또는 ZIP 출력을 위한 임시 파일 생성
        base = f"enhanced_av_asr_{int(os.getpid())}"
        
        if format == "srt":
            # SRT 생성
            srt_segments = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result["segments"]
            ]
            
            srt_text = segments_to_srt(srt_segments)
            srt_path = f"{base}.srt"
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
            
            return FileResponse(
                srt_path, 
                media_type="application/x-subrip", 
                filename="captions_enhanced.srt"
            )
        
        if format == "both":
            # JSON과 SRT 모두 생성
            json_path = f"{base}_result.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # SRT 생성
            srt_segments = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result["segments"]
            ]
            
            srt_text = segments_to_srt(srt_segments)
            srt_path = f"{base}_captions.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
            
            # ZIP 파일 생성
            zip_path = f"{base}_result.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(srt_path, arcname="captions_enhanced.srt")
                z.write(json_path, arcname="result_enhanced.json")
            
            return FileResponse(
                zip_path, 
                media_type="application/zip", 
                filename="enhanced_av_asr_result.zip"
            )
        
    except Exception as e:
        # 임시 파일 정리
        try:
            os.unlink(src_path)
        except:
            pass
        
        error_msg = f"Enhanced AV-ASR 추론 실패: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


# SRT 유틸리티 import
from server.utils.srt import segments_to_srt


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 처리"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An unexpected error occurred",
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    # 개발 서버 실행
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
