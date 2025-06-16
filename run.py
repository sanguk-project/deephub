#!/usr/bin/env python3
"""
ILJoo Deep Hub - 사용자 서비스 시작
새로 정리된 폴더 구조에서 RAG 서비스를 시작합니다.
"""

import multiprocessing
import atexit
import signal

# 시작 최적화 먼저 적용
from startup_optimization import optimize_startup
optimize_startup()

from shared.utils import setup_environment
from shared.config.settings import settings

def cleanup_resources():
    """프로세스 종료 시 리소스 정리"""
    if multiprocessing.current_process().name == 'MainProcess':
        for process in multiprocessing.active_children():
            process.terminate()
            process.join()

# 종료 시 리소스 정리 등록
atexit.register(cleanup_resources)
signal.signal(signal.SIGTERM, lambda signum, frame: cleanup_resources())
signal.signal(signal.SIGINT, lambda signum, frame: cleanup_resources())

async def check_vector_config_and_reindex():
    """VectorDB 설정 변경 확인 및 자동 재적재"""
    try:
        from service.storage.vector_config_manager import auto_check_and_update
        
        print("🔍 VectorDB 적재 조건 변경 확인...")
        
        # 설정 변경 확인 및 자동 업데이트
        updated = await auto_check_and_update()
        
        if updated:
            print("🔄 VectorDB 적재 조건 변경으로 인한 자동 재적재 완료!")
        else:
            print("✅ VectorDB 적재 조건 변경 없음")
            
    except Exception as e:
        print(f"⚠️  VectorDB 설정 확인 중 오류: {e}")
        print("📝 수동으로 재적재가 필요할 수 있습니다.")

async def auto_index_documents():
    """백그라운드에서 documents 폴더의 새로운 파일들을 자동 인덱싱"""
    try:
        from admin.tools.document_indexer import auto_index_documents
        
        print("📁 문서 자동 인덱싱 시작...")
        result = await auto_index_documents()
        
        if result.get("indexed_files", 0) > 0:
            print(f"✅ 새 문서 {result['indexed_files']}개 인덱싱 완료")
        
        if result.get("failed_files", 0) > 0:
            print(f"⚠️  오류 발생: {result['failed_files']}개 파일")
        
        if result.get("total_files", 0) == 0:
            print("📂 documents 폴더가 비어있거나 지원되는 파일이 없습니다")
            print(f"📍 경로: {settings.path.documents_dir}")
            print("💡 지원 형식: .pdf, .txt, .docx, .md, .html, .rtf")
        else:
            print(f"📊 전체 파일: {result['total_files']}개, 인덱싱: {result['indexed_files']}개")
        
    except Exception as e:
        print(f"⚠️  자동 인덱싱 중 오류: {e}")

def main():
    """서비스 시작 (빠른 시작 최적화)"""
    # 멀티프로세싱 시작 메서드 설정
    multiprocessing.set_start_method('spawn', force=True)
    
    # 기본 환경 설정만 수행 (무거운 모델 로딩 제외)
    setup_environment()
    
    print("🚀 ILJoo Deep Hub RAG Service (Fast Start)")
    print("=" * 50)
    
    # 필수 검증만 수행 (API 키 위주)
    api_key_valid = bool(settings.api.openai_api_key)
    
    if not api_key_valid:
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("\n🔧 OpenAI API 키 설정 방법:")
        print("1. 터미널에서 환경변수 설정:")
        print("   export OPENAI_API_KEY=\"your-api-key-here\"")
        print("\n2. 영구 설정 (권장):")
        print("   echo 'export OPENAI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc")
        print("   source ~/.bashrc")
        print("\n3. 현재 세션에서만 실행:")
        print("   OPENAI_API_KEY=\"your-api-key-here\" python run.py")
        print("\n💡 OpenAI API 키는 https://platform.openai.com/api-keys 에서 발급받으실 수 있습니다.")
        return
    
    print("✅ 기본 설정 검증 완료")
    print("⚡ 빠른 시작 모드: 모델들은 첫 요청 시에 로딩됩니다")
    
    # VectorDB 설정 변경 확인 및 자동 재적재
    import asyncio
    asyncio.run(check_vector_config_and_reindex())
    
    # 자동 문서 인덱싱 (동기 처리)
    asyncio.run(auto_index_documents())
    
    print("🎯 서버 시작 중...")
    
    # FastAPI 서버 실행
    import uvicorn
    
    print(f"🌐 서버 주소: http://{settings.server.host}:{settings.server.port}")
    print("📖 API 문서: http://localhost:9999/docs")
    print("🔧 시스템 상태: http://localhost:9999/status")
    print("💬 메인 화면: http://localhost:9999/")
    print("\n📝 성능 팁:")
    print("  - 첫 질문 시 모델 로딩으로 약간의 시간이 소요됩니다")
    print("  - 이후부터는 빠른 응답이 가능합니다")
    print("\n📁 문서 관리:")
    print(f"  - 새 문서를 '{settings.path.documents_dir}' 폴더에 추가하세요")
    print("  - 재시작 시 자동으로 새 문서가 인덱싱됩니다")
    print("  - 수동 인덱싱: python admin.py")
    
    # 항상 reload 모드 사용 (개발 편의성)
    uvicorn.run(
        "service.api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level=settings.server.log_level,
        access_log=False  # 로그 최적화
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_resources()
    except Exception as e:
        print(f"오류 발생: {e}")
        cleanup_resources() 