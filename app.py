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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 전역 변수
models_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global models_loaded
    
    # 시작 시 모델 로드
    logger.info("🚀 Starting Enhanced AV-ASR Server...")
    
    try:
        # 모델 로드 테스트
        from server.pipeline.ec_integration_pipeline import infer_media_for_ec
        models_loaded = True
        
        logger.info("✅ Enhanced AV-ASR Server started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        models_loaded = False
    
    yield
    
    # 종료 시 정리
    logger.info("🛑 Shutting down Enhanced AV-ASR Server...")


# FastAPI 앱 생성
app = FastAPI(
    title="Enhanced AV-ASR System for EC Model Integration",
    description="""
    Enhanced AV-ASR (Audio-Visual Automatic Speech Recognition) 시스템
    
    ## 주요 특징
    - **Wav2Vec2 + Whisper 앙상블**: 고품질 오디오 인코딩
    - **Enhanced CTC Decoder**: GELU 활성화, Beam Search 최적화
    - **25fps 적응형 키워드 생성**: 신뢰도 기반 자동 키워드 추출
    - **EC 모델 연동**: 토큰/단어별 상세 정보, n-best 후보
    - **한국어 특화**: 한국어 후처리 및 신뢰도 기반 필터링
    - **실시간 처리**: 25fps 프레임 기반 윈도우 처리
    
    ## API 엔드포인트
    - `POST /v1/enhanced_infer`: Enhanced AV-ASR 추론 (EC 모델 연동용)
    - `GET /v1/enhanced_info`: 시스템 정보
    - `GET /v1/health`: 헬스 체크
    - `POST /v1/srt`: SRT 자막 생성
    """,
    version="0.9.4",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    summary="루트 엔드포인트",
    description="Enhanced AV-ASR 시스템의 기본 정보를 반환합니다."
)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Enhanced AV-ASR System for EC Model Integration",
        "version": "0.9.4",
        "status": "running",
        "docs": "/docs",
        "health": "/v1/health",
        "features": [
            "Wav2Vec2 + Whisper 앙상블",
            "Enhanced CTC Decoder with GELU",
            "25fps 적응형 자동 키워드 생성",
            "EC 모델 연동용 상세 출력",
            "한국어 특화 후처리",
            "Swagger UI 성능 최적화"
        ]
    }


@app.get(
    "/v1/health",
    summary="헬스 체크",
    description="시스템 상태와 모델 로드 상태를 확인합니다."
)
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        global models_loaded
        
        # 모델 상태 확인
        model_status = {
            "wav2vec2": models_loaded,
            "whisper": models_loaded,
            "enhanced_ctc_decoder": models_loaded,
            "ec_integration_pipeline": models_loaded
        }
        
        # 전체 상태 결정
        all_models_ready = all(model_status.values())
        
        health_info = {
            "status": "ok" if all_models_ready else "degraded",
            "models": model_status,
            "device": "cpu",  # TODO: GPU 감지 로직 추가
            "fps": 25,
            "pipeline": "Enhanced AV-ASR: Wav2Vec2 + Whisper → Enhanced CTC + 25fps 적응형 키워드 생성",
            "ec_model_ready": all_models_ready,
            "swagger_optimized": True,
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
        
        status_code = 200 if all_models_ready else 503
        return JSONResponse(content=health_info, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "models": {"wav2vec2": False, "whisper": False, "enhanced_ctc_decoder": False},
                "ec_model_ready": False
            },
            status_code=500
        )


@app.get(
    "/v1/enhanced_info",
    summary="Enhanced AV-ASR 시스템 정보",
    description="Enhanced AV-ASR 시스템의 모델 정보와 설정을 반환합니다."
)
async def get_enhanced_info():
    """Enhanced AV-ASR 시스템 정보를 반환합니다."""
    
    try:
        info = {
            "system": "Enhanced AV-ASR System for EC Model Integration",
            "version": "0.9.4",
            "features": [
                "Wav2Vec2 + Whisper 앙상블",
                "Enhanced CTC Decoder with GELU",
                "25fps 적응형 자동 키워드 생성",
                "Korean-specific post-processing",
                "Confidence-based filtering",
                "Beam search optimization (beam_size=5)",
                "EC Model Integration Ready"
            ],
            "default_settings": {
                "beam_size": 5,
                "confidence_threshold": 0.01,
                "audio_fusion_method": "weighted",
                "audio_fusion_alpha": 0.6,
                "lm_weight": 0.6,
                "chunk_ms": 400,
                "fps": 25
            },
            "supported_formats": ["mp4", "avi", "mov", "mkv", "wav", "m4a"],
            "supported_languages": ["ko", "en"],
            "output_formats": ["json", "srt", "both"],
            "ec_model_ready": True,
            "audio_fusion_methods": [
                "weighted", "max", "adaptive", "concat"
            ]
        }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        error_msg = f"시스템 정보 조회 실패: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.post(
    "/v1/enhanced_infer",
    summary="Enhanced AV-ASR → EC 모델 연동용 자막 생성 (25fps 자동 키워드 생성)",
    description="""
    Enhanced AV-ASR 파이프라인을 사용하여 EC 모델과 연동할 수 있는 상세한 출력을 생성합니다.
    
    주요 특징:
    - Wav2Vec2 + Whisper 앙상블
    - 토큰별 logprob, 타임스탬프, 프레임 엔트로피
    - n-best 후보 및 스코어
    - 한국어 특화 후처리
    - 신뢰도 기반 필터링
    - 25fps 적응형 자동 키워드 생성
    """,
    responses={
        200: {
            "description": "EC 모델 연동용 상세 JSON",
            "content": {
                "application/json": {"schema": {"type": "object"}},
                "application/x-subrip": {"schema": {"type": "string", "format": "binary"}},
                "application/zip": {"schema": {"type": "string", "format": "binary"}}
            }
        },
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
    
    # 4) Enhanced AV-ASR 추론 (기존 코드 사용)
    try:
        from server.pipeline.ec_integration_pipeline import infer_media_for_ec
        
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
