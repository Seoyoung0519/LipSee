"""
런타임 환경설정 모듈
- KoBART 체크포인트 경로, 마이크로배칭 문자수 상한, 세그먼트 경계 토큰 등
"""
import os

# KoBART 교정용 체크포인트 경로(HF repo 이름 또는 로컬 디렉토리)
KOBART_EC_PATH = os.getenv("KOBART_EC_PATH", "your/kobart-ec-ckpt")

# 합본 인퍼런스 시 1회 처리 최대 문자수(문맥 확보 + 속도 균형)
CHUNK_CHAR_LIMIT = int(os.getenv("CHUNK_CHAR_LIMIT", "1200"))

# 세그먼트 경계 토큰: 합본 인퍼런스 후 결과를 같은 위치에서 split하기 위한 마커
SEG_TOKEN = os.getenv("SEG_TOKEN", "<SEG>")