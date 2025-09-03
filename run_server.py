#!/usr/bin/env python3
"""
Enhanced AV-ASR FastAPI Server 실행 스크립트
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def check_dependencies():
    """의존성 확인"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'librosa',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_server_structure():
    """server 폴더 구조 확인"""
    print("🔍 Checking server structure...")
    
    required_files = [
        "server/pipeline/ec_integration_pipeline.py",
        "server/models/wav2vec2_encoder.py",
        "server/models/whisper_encoder.py",
        "server/models/config.py",
        "app.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("   Please ensure all server files are present")
        return False
    
    print("✅ Server structure is correct")
    return True

def start_server():
    """서버 시작"""
    print("🚀 Starting Enhanced AV-ASR Server...")
    print("=" * 50)
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    try:
        # uvicorn으로 서버 실행
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        print(f"   URL: http://localhost:8000")
        print(f"   Docs: http://localhost:8000/docs")
        print("=" * 50)
        
        # 서버 프로세스 시작
        process = subprocess.Popen(cmd, env=env)
        
        # 서버가 시작될 때까지 대기
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # 서버 상태 확인
        try:
            import requests
            response = requests.get("http://localhost:8000/v1/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                print("🎉 Enhanced AV-ASR Server is running")
                print("\n📋 Available endpoints:")
                print("   - GET  /                    - Root endpoint")
                print("   - GET  /v1/health          - Health check")
                print("   - GET  /v1/enhanced_info   - System info")
                print("   - POST /v1/enhanced_infer  - Enhanced AV-ASR inference")
                print("   - GET  /docs               - Swagger UI")
                print("   - GET  /redoc              - ReDoc")
            else:
                print(f"⚠️  Server started but health check failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not verify server status: {e}")
        
        print("\n🛑 Press Ctrl+C to stop the server")
        
        # 서버 프로세스 대기
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def main():
    """메인 함수"""
    print("🎯 Enhanced AV-ASR FastAPI Server Launcher")
    print("=" * 50)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        return False
    
    print()
    
    # Server 구조 확인
    if not check_server_structure():
        print("\n❌ Server structure check failed")
        return False
    
    print()
    
    # 서버 시작
    return start_server()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
