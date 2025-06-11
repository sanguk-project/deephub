"""
ILJoo Deep Hub - 공용 유틸리티
중복되는 기능들을 통합한 유틸리티 모듈
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import openai
from openai import AsyncOpenAI

from .config.settings import settings

logger = logging.getLogger(__name__)

class ClientManager:
    """OpenAI 클라이언트 관리자 - 싱글톤 패턴"""
    
    _sync_client: Optional[openai.OpenAI] = None
    _async_client: Optional[AsyncOpenAI] = None
    
    @classmethod
    def get_sync_client(cls) -> openai.OpenAI:
        """동기 OpenAI 클라이언트 반환 (싱글톤)"""
        if cls._sync_client is None:
            if not settings.api.openai_api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
            cls._sync_client = openai.OpenAI(api_key=settings.api.openai_api_key)
            logger.info("OpenAI 동기 클라이언트 초기화 완료")
        
        return cls._sync_client
    
    @classmethod
    def get_async_client(cls) -> AsyncOpenAI:
        """비동기 OpenAI 클라이언트 반환 (싱글톤)"""
        if cls._async_client is None:
            if not settings.api.openai_api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
            cls._async_client = AsyncOpenAI(api_key=settings.api.openai_api_key)
            logger.info("OpenAI 비동기 클라이언트 초기화 완료")
        
        return cls._async_client
    
    @classmethod
    def reset_clients(cls):
        """클라이언트 재설정 (테스트용)"""
        cls._sync_client = None
        cls._async_client = None

class PathManager:
    """경로 관리자"""
    
    @staticmethod
    def get_project_root() -> Path:
        """프로젝트 루트 경로 반환"""
        return Path(__file__).parent.parent
    
    @staticmethod
    def setup_python_path():
        """Python 경로 설정"""
        import sys
        project_root = PathManager.get_project_root()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    
    @staticmethod
    def ensure_directories():
        """필요한 디렉토리들을 생성"""
        directories = [
            settings.path.documents_dir,
            settings.path.faiss_index_dir,
            settings.path.templates_dir,
            settings.path.static_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("필요한 디렉토리들을 생성했습니다.")

class LoggingManager:
    """로깅 관리자"""
    
    @staticmethod
    def setup_logging(level: str = "INFO") -> logging.Logger:
        """표준 로깅 설정"""
        # 기본 로깅 설정
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 특정 라이브러리 로그 레벨 조정
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.info("로깅 시스템 초기화 완료")
        
        return logger

class ValidationManager:
    """검증 관리자"""
    
    @staticmethod
    def validate_api_keys() -> Dict[str, bool]:
        """API 키 유효성 검사"""
        validation_results = {
            "openai_api_key": bool(settings.api.openai_api_key),
        }
        
        return validation_results
    
    @staticmethod
    def validate_paths() -> Dict[str, bool]:
        """경로 유효성 검사"""
        validation_results = {
            "documents_dir": os.path.exists(settings.path.documents_dir),
            "faiss_index_dir": os.path.exists(settings.path.faiss_index_dir),
        }
        
        return validation_results
    
    @staticmethod
    def validate_system() -> Dict[str, Any]:
        """시스템 전체 검증"""
        api_validation = ValidationManager.validate_api_keys()
        path_validation = ValidationManager.validate_paths()
        
        return {
            "api_keys": api_validation,
            "paths": path_validation,
            "all_valid": all(api_validation.values()) and all(path_validation.values())
        }

def get_openai_client(async_mode: bool = False):
    """OpenAI 클라이언트 반환 (편의 함수)"""
    if async_mode:
        return ClientManager.get_async_client()
    else:
        return ClientManager.get_sync_client()

def setup_environment():
    """환경 초기 설정"""
    # Python 경로 설정
    PathManager.setup_python_path()
    
    # 디렉토리 생성
    PathManager.ensure_directories()
    
    # 로깅 설정
    LoggingManager.setup_logging()
    
    logger.info("환경 설정 완료")

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 반환"""
    validation = ValidationManager.validate_system()
    
    return {
        "project_root": str(PathManager.get_project_root()),
        "settings": {
            "models": {
                "embedding": settings.model.embedding_model,
                "verification": settings.model.verification_model,
                "final_answer": settings.model.final_answer_model,
            },
            "server": {
                "host": settings.server.host,
                "port": settings.server.port,
            }
        },
        "validation": validation,
        "version": "2.0.0"
    } 