"""
ILJoo Deep Hub 시스템 설정
환경변수와 기본 설정값들을 관리합니다.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    database_name: str = "rag_system"

@dataclass
class APIConfig:
    """API 키 설정
    
    ⚠️ 중요: OPENAI_API_KEY 환경변수를 설정해야 합니다.
    
    설정 방법:
    1. 터미널에서: export OPENAI_API_KEY="your-api-key-here"
    2. ~/.bashrc에 추가: echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
    3. 또는 아래 openai_api_key 값을 직접 수정 (보안상 권장하지 않음)
    """
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

@dataclass
class ModelConfig:
    """모델 설정"""
    embedding_model: str = "BAAI/bge-m3"
    verification_model: str = "gpt-4.1-2025-04-14"
    final_answer_model: str = "gpt-4.1-2025-04-14"
    exaone_model_path: str = "/mnt/ssd/1/hub/EXAONE-3.5-2.4B-Instruct"  # 로컬 저장 경로
    exaone_model_name: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"     # HuggingFace 모델명

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    max_retrieved_docs: int = 8  # 적절한 수의 문서 검색 (품질과 성능 균형)
    similarity_threshold: float = 0.3  # 더 엄격한 임계값으로 품질 향상
    max_context_length: int = 8000  # 더 많은 컨텍스트 허용
    min_score_threshold: int = 6  # 품질 임계값 강화 (6점 이상)
    max_retry_attempts: int = 3  # 적절한 재시도 횟수
    
    # 신뢰도 향상을 위한 추가 설정
    keyword_weight: float = 0.3  # 키워드 매칭 가중치
    semantic_weight: float = 0.7  # 의미적 유사도 가중치
    diversity_threshold: int = 3  # 같은 소스에서 최대 문서 수
    min_text_length: int = 30  # 최소 텍스트 길이
    context_relevance_threshold: float = 0.6  # 컨텍스트 관련성 임계값

@dataclass
class PathConfig:
    """경로 설정"""
    documents_dir: str = str(PROJECT_ROOT / "shared" / "data" / "documents")
    faiss_index_dir: str = str(PROJECT_ROOT / "shared" / "data" / "faiss_index")
    templates_dir: str = str(PROJECT_ROOT / "service" / "web" / "template")
    static_dir: str = str(PROJECT_ROOT / "service" / "web" / "static")

@dataclass
class ServerConfig:
    """서버 설정"""
    host: str = "0.0.0.0"
    port: int = 9999
    reload: bool = True
    log_level: str = "info"

class Settings:
    """전체 설정 클래스"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.model = ModelConfig()
        self.rag = RAGConfig()
        self.path = PathConfig()
        self.server = ServerConfig()
    


# 전역 설정 인스턴스
settings = Settings() 