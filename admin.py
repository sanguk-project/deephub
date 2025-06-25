#!/usr/bin/env python3
"""
ILJoo Deep Hub - 관리자 도구
VectorDB 관리, 문서 인덱싱, 설정 확인 등의 관리 기능을 제공
"""

import asyncio
import argparse
from typing import Dict, Any

from startup_optimization import optimize_startup
optimize_startup()

from shared.utils import setup_environment
from shared.config.settings import settings

def print_banner():
    """관리자 도구 배너 출력"""
    print("🛠️  ILJoo Deep Hub - 관리자 도구")
    print("=" * 50)

async def check_vector_config():
    """VectorDB 설정 상태 확인"""
    try:
        from service.storage.vector_config_manager import get_config_status
        
        print("🔍 VectorDB 설정 상태 확인 중...")
        status = get_config_status()
        
        print("\n📊 현재 설정:")
        config = status['current_config']
        print(f"  - Chunk Size: {config['chunk_size']}")
        print(f"  - Chunk Overlap: {config['chunk_overlap']}")
        print(f"  - Embedding Model: {config['embedding_model']}")
        print(f"  - Max Retrieved Docs: {config['max_retrieved_docs']}")
        print(f"  - Similarity Threshold: {config['similarity_threshold']}")
        print(f"  - Context Length: {config['max_context_length']}")
        
        print(f"\n📁 경로 정보:")
        print(f"  - Documents Dir: {settings.path.documents_dir}")
        print(f"  - FAISS Index Dir: {settings.path.faiss_index_dir}")
        print(f"  - Config File Exists: {status['config_file_exists']}")
        print(f"  - Documents Dir Exists: {status['documents_dir_exists']}")
        
        if status['config_changed']:
            print("\n⚠️  설정 변경이 감지되었습니다!")
            print("   재적재가 필요할 수 있습니다.")
        else:
            print("\n✅ 설정 변경 없음")
            
    except Exception as e:
        print(f"❌ 설정 확인 중 오류: {e}")

async def force_reindex():
    """강제 VectorDB 초기화 및 재적재"""
    try:
        from service.storage.vector_config_manager import force_reset_and_reindex
        
        print("🔄 VectorDB 강제 초기화 및 재적재 시작...")
        print("⚠️  기존 벡터 데이터가 모두 삭제됩니다!")
        
        # 사용자 확인
        response = input("계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("❌ 작업이 취소되었습니다.")
            return
        
        await force_reset_and_reindex()
        print("✅ VectorDB 강제 재적재 완료!")
        
    except Exception as e:
        print(f"❌ 강제 재적재 중 오류: {e}")

async def auto_check_and_reindex():
    """자동 설정 확인 및 필요시 재적재"""
    try:
        from service.storage.vector_config_manager import auto_check_and_update
        
        print("🔍 설정 변경 확인 및 자동 재적재...")
        
        updated = await auto_check_and_update()
        
        if updated:
            print("✅ 설정 변경으로 인한 자동 재적재 완료!")
        else:
            print("✅ 설정 변경 없음 - 재적재 불필요")
            
    except Exception as e:
        print(f"❌ 자동 재적재 중 오류: {e}")

async def manual_index_documents():
    """수동 문서 인덱싱"""
    try:
        from admin.tools.document_indexer import auto_index_documents
        
        print("📁 수동 문서 인덱싱 시작...")
        result = await auto_index_documents()
        
        print(f"\n📊 인덱싱 결과:")
        print(f"  - 전체 파일: {result.get('total_files', 0)}개")
        print(f"  - 인덱싱 성공: {result.get('indexed_count', 0)}개")
        print(f"  - 인덱싱 실패: {result.get('failed_count', 0)}개")
        
        if result.get('failed_files'):
            print(f"\n❌ 실패한 파일들:")
            for file_path in result['failed_files']:
                print(f"  - {file_path}")
                
    except Exception as e:
        print(f"❌ 수동 인덱싱 중 오류: {e}")

async def get_vector_store_info():
    """VectorDB 상태 정보 조회"""
    try:
        from service.storage.vector_store import get_vector_store_info
        
        print("📊 VectorDB 상태 정보 조회 중...")
        info = get_vector_store_info()
        
        print(f"\n📈 VectorDB 정보:")
        print(f"  - 총 문서 수: {info.get('total_documents', 0)}개")
        print(f"  - 인덱스 크기: {info.get('index_size', 0)} bytes")
        print(f"  - 마지막 업데이트: {info.get('last_updated', 'N/A')}")
        
    except Exception as e:
        print(f"❌ VectorDB 정보 조회 중 오류: {e}")

def show_help():
    """도움말 출력"""
    print("""
    🛠️  사용 가능한 명령어:

    1. 설정 관리:
    python admin.py --check-config     # VectorDB 설정 상태 확인
    python admin.py --auto-reindex     # 자동 설정 확인 및 재적재
    python admin.py --force-reindex    # 강제 초기화 및 재적재

    2. 문서 관리:
    python admin.py --index-docs       # 수동 문서 인덱싱
    python admin.py --vector-info      # VectorDB 상태 정보

    3. 기타:
    python admin.py --help             # 이 도움말 출력

    💡 팁:
    - 설정 변경 후에는 --auto-reindex를 실행하세요
    - 문제가 있을 때는 --force-reindex로 완전 초기화하세요
    - 새 문서 추가 후에는 --index-docs로 수동 인덱싱하세요
""")

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="ILJoo Deep Hub 관리자 도구")
    parser.add_argument("--check-config", action="store_true", help="VectorDB 설정 상태 확인")
    parser.add_argument("--auto-reindex", action="store_true", help="자동 설정 확인 및 재적재")
    parser.add_argument("--force-reindex", action="store_true", help="강제 초기화 및 재적재")
    parser.add_argument("--index-docs", action="store_true", help="수동 문서 인덱싱")
    parser.add_argument("--vector-info", action="store_true", help="VectorDB 상태 정보")
    parser.add_argument("--help-admin", action="store_true", help="관리자 도구 도움말")
    
    args = parser.parse_args()
    
    # 환경 설정
    setup_environment()
    print_banner()
    
    # 명령어 실행
    if args.check_config:
        await check_vector_config()
    elif args.auto_reindex:
        await auto_check_and_reindex()
    elif args.force_reindex:
        await force_reindex()
    elif args.index_docs:
        await manual_index_documents()
    elif args.vector_info:
        await get_vector_store_info()
    elif args.help_admin:
        show_help()
    else:
        # 기본 대화형 메뉴
        await interactive_menu()

async def interactive_menu():
    """대화형 메뉴"""
    while True:
        print("\n🛠️  관리자 도구 메뉴:")
        print("1. VectorDB 설정 상태 확인")
        print("2. 자동 설정 확인 및 재적재")
        print("3. 강제 초기화 및 재적재")
        print("4. 수동 문서 인덱싱")
        print("5. VectorDB 상태 정보")
        print("6. 도움말")
        print("0. 종료")
        
        try:
            choice = input("\n선택하세요 (0-6): ").strip()
            
            if choice == "1":
                await check_vector_config()
            elif choice == "2":
                await auto_check_and_reindex()
            elif choice == "3":
                await force_reindex()
            elif choice == "4":
                await manual_index_documents()
            elif choice == "5":
                await get_vector_store_info()
            elif choice == "6":
                show_help()
            elif choice == "0":
                print("👋 관리자 도구를 종료합니다.")
                break
            else:
                print("❌ 올바른 번호를 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 관리자 도구를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 관리자 도구를 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 관리자 도구 실행 중 오류: {e}")
        sys.exit(1) 