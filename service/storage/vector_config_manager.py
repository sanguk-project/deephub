"""
VectorDB 적재 조건 관리자
적재 조건 변경 감지 및 자동 초기화/재적재 기능을 제공
"""

import os
import json
import hashlib
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from shared.config.settings import settings
from admin.tools.document_indexer import IndexingConfig, DocumentIndexer
from service.storage.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

class VectorConfigManager:
    """VectorDB 적재 조건 관리자"""
    
    def __init__(self):
        self.config_file = os.path.join(settings.path.faiss_index_dir, "vector_config.json")
        self.documents_dir = settings.path.documents_dir
        self.current_config = self._get_current_config()
        
    def _get_current_config(self) -> Dict[str, Any]:
        """현재 적재 조건 구성을 가져옴"""
        indexing_config = IndexingConfig()
        
        return {
            "chunk_size": indexing_config.chunk_size,
            "chunk_overlap": indexing_config.chunk_overlap,
            "embedding_model": settings.model.embedding_model,
            "max_retrieved_docs": settings.rag.max_retrieved_docs,
            "similarity_threshold": settings.rag.similarity_threshold,
            "max_context_length": settings.rag.max_context_length,
            "keyword_weight": settings.rag.keyword_weight,
            "semantic_weight": settings.rag.semantic_weight,
            "diversity_threshold": settings.rag.diversity_threshold,
            "min_text_length": settings.rag.min_text_length,
            "context_relevance_threshold": settings.rag.context_relevance_threshold,
            "supported_formats": indexing_config.supported_formats,
            "config_version": "1.0",
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """설정 해시 계산 (버전과 타임스탬프 제외)"""
        # 버전과 타임스탬프를 제외한 설정으로 해시 계산
        config_for_hash = {k: v for k, v in config.items() 
                          if k not in ["config_version", "last_updated"]}
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_saved_config(self) -> Optional[Dict[str, Any]]:
        """저장된 설정 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"저장된 설정 로드 실패: {e}")
        return None
    
    def _save_config(self, config: Dict[str, Any]):
        """현재 설정 저장"""
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"VectorDB 설정 저장 완료: {self.config_file}")
            
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
    
    def check_config_changed(self) -> bool:
        """적재 조건 변경 여부 확인"""
        try:
            saved_config = self._load_saved_config()
            
            # 저장된 설정이 없으면 첫 실행으로 간주
            if saved_config is None:
                logger.info("저장된 VectorDB 설정이 없음 - 첫 실행")
                return True
            
            # 설정 해시 비교
            current_hash = self._calculate_config_hash(self.current_config)
            saved_hash = self._calculate_config_hash(saved_config)
            
            if current_hash != saved_hash:
                logger.info("VectorDB 적재 조건 변경 감지!")
                logger.info("변경된 설정:")
                
                # 변경사항 로깅
                for key, current_value in self.current_config.items():
                    if key in ["config_version", "last_updated"]:
                        continue
                    saved_value = saved_config.get(key)
                    if current_value != saved_value:
                        logger.info(f"  {key}: {saved_value} -> {current_value}")
                
                return True
            
            logger.info("VectorDB 적재 조건 변경 없음")
            return False
            
        except Exception as e:
            logger.error(f"설정 변경 확인 중 오류: {e}")
            # 오류 시 안전하게 재적재
            return True
    
    async def reset_vector_db(self):
        """VectorDB 완전 초기화"""
        try:
            logger.info("VectorDB 완전 초기화 시작...")
            
            # 1. 기존 글로벌 인스턴스 정리
            try:
                from service.storage.vector_store import faiss_store
                if hasattr(faiss_store, 'delete_collection'):
                    faiss_store.delete_collection()
                    logger.info("기존 VectorStore 인스턴스 정리 완료")
            except Exception as e:
                logger.warning(f"기존 인스턴스 정리 중 오류: {e}")
            
            # 2. FAISS 인덱스 파일들 완전 삭제
            faiss_dir = settings.path.faiss_index_dir
            if os.path.exists(faiss_dir):
                deleted_files = []
                for file_name in os.listdir(faiss_dir):
                    # 모든 FAISS 관련 파일 삭제 (.faiss, .pkl, .json 등)
                    if any(file_name.endswith(ext) for ext in ['.faiss', '.pkl', '.json', '.bin', '.index']):
                        file_path = os.path.join(faiss_dir, file_name)
                        try:
                            os.remove(file_path)
                            deleted_files.append(file_name)
                            logger.debug(f"파일 삭제: {file_name}")
                        except Exception as e:
                            logger.warning(f"파일 삭제 실패 {file_path}: {e}")
                
                if deleted_files:
                    logger.info(f"삭제된 파일들: {', '.join(deleted_files)}")
                else:
                    logger.info("삭제할 FAISS 파일이 없음")
            else:
                logger.info(f"FAISS 인덱스 디렉토리가 존재하지 않음: {faiss_dir}")
            
            # 3. 디렉토리 재생성 (완전히 새로 만들기)
            try:
                if os.path.exists(faiss_dir):
                    import shutil
                    shutil.rmtree(faiss_dir)
                    logger.info(f"FAISS 디렉토리 완전 삭제: {faiss_dir}")
                
                os.makedirs(faiss_dir, exist_ok=True)
                logger.info(f"FAISS 디렉토리 재생성: {faiss_dir}")
            except Exception as e:
                logger.warning(f"디렉토리 재생성 중 오류: {e}")
                os.makedirs(faiss_dir, exist_ok=True)
            
            # 4. 새 VectorStore 인스턴스로 글로벌 변수 교체
            try:
                import importlib
                import service.storage.vector_store as vs_module
                importlib.reload(vs_module)  # 모듈 재로드로 글로벌 인스턴스 재생성
                logger.info("VectorStore 모듈 재로드 완료")
            except Exception as e:
                logger.warning(f"모듈 재로드 실패, 새 인스턴스 생성: {e}")
                # 재로드 실패 시 새 인스턴스 생성
                from service.storage.vector_store import FAISSVectorStore
                new_store = FAISSVectorStore()
            
            logger.info("VectorDB 완전 초기화 완료")
            
        except Exception as e:
            logger.error(f"VectorDB 초기화 중 오류: {e}")
            raise e
    
    async def reindex_all_documents(self):
        """모든 문서 재적재"""
        try:
            logger.info("문서 재적재 시작...")
            
            if not os.path.exists(self.documents_dir):
                logger.warning(f"문서 디렉토리가 존재하지 않음: {self.documents_dir}")
                return
            
            # 새 인덱서 생성
            indexer = DocumentIndexer()
            
            # 디렉토리 내 모든 문서 인덱싱
            result = await indexer.index_directory(
                directory_path=self.documents_dir,
                recursive=True
            )
            
            if result.get("success", False):
                indexed_count = result.get("indexed_files", 0)
                failed_count = result.get("failed_files", 0)
                logger.info(f"문서 재적재 완료 - 성공: {indexed_count}개, 실패: {failed_count}개")
            else:
                logger.error(f"문서 재적재 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"문서 재적재 중 오류: {e}")
            raise e
    
    async def handle_config_change(self):
        """설정 변경 처리 (초기화 + 재적재)"""
        try:
            logger.info("VectorDB 설정 변경 처리 시작...")
            
            # 1. VectorDB 초기화
            await self.reset_vector_db()
            
            # 2. 모든 문서 재적재
            await self.reindex_all_documents()
            
            # 3. 현재 설정 저장
            self._save_config(self.current_config)
            
            logger.info("VectorDB 설정 변경 처리 완료!")
            
        except Exception as e:
            logger.error(f"설정 변경 처리 중 오류: {e}")
            raise e
    
    async def check_and_update_if_needed(self):
        """필요시 자동 업데이트 실행"""
        try:
            if self.check_config_changed():
                logger.info("VectorDB 적재 조건 변경으로 인한 자동 업데이트 시작...")
                await self.handle_config_change()
                return True
            return False
            
        except Exception as e:
            logger.error(f"자동 업데이트 중 오류: {e}")
            raise e
    
    def get_current_config_info(self) -> Dict[str, Any]:
        """현재 설정 정보 반환"""
        config = self.current_config.copy()
        config["config_hash"] = self._calculate_config_hash(config)
        return config
    
    def get_config_status(self) -> Dict[str, Any]:
        """설정 상태 정보 반환"""
        saved_config = self._load_saved_config()
        current_config = self.current_config
        
        return {
            "current_config": current_config,
            "saved_config": saved_config,
            "config_changed": self.check_config_changed(),
            "config_file_exists": os.path.exists(self.config_file),
            "documents_dir_exists": os.path.exists(self.documents_dir),
            "faiss_index_dir": settings.path.faiss_index_dir
        }


# 전역 인스턴스
_config_manager = None

def get_vector_config_manager() -> VectorConfigManager:
    """VectorConfigManager 싱글톤 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = VectorConfigManager()
    return _config_manager

async def auto_check_and_update():
    """자동 설정 확인 및 업데이트"""
    manager = get_vector_config_manager()
    return await manager.check_and_update_if_needed()

async def force_reset_and_reindex():
    """강제 초기화 및 재적재"""
    manager = get_vector_config_manager()
    await manager.handle_config_change()

def get_config_status():
    """설정 상태 정보 조회"""
    manager = get_vector_config_manager()
    return manager.get_config_status() 