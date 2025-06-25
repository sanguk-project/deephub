"""
ILJoo Deep Hub 시스템 설정
환경변수와 기본 설정값들을 관리합니다.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    database_name: str = "rag_system"

@dataclass
class APIConfig:
    """API 키 설정
    
    ⚠️ 보안 중요사항:
    - API 키는 절대 코드에 직접 작성하지 마세요
    - 반드시 환경변수로만 설정하세요
    - .env 파일 사용 시 .gitignore에 포함하세요
    
    설정 방법:
    1. .env 파일 생성: cp .env.example .env
    2. .env 파일에서 실제 API 키 설정
    3. 또는 터미널에서: export OPENAI_API_KEY="your-actual-api-key"
    4. ~/.bashrc 영구 설정: echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
    
    API 키 발급:
    - OpenAI: https://platform.openai.com/api-keys
    - Google: https://console.cloud.google.com/apis/credentials
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
    
    # Re-ranker 모델 설정 (Cross Encoder)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross Encoder 모델
    reranker_model_type: str = "cross_encoder"  # 모델 타입
    reranker_device: str = "auto"  # 디바이스 설정 (auto, cpu, cuda)
    reranker_max_length: int = 512  # 최대 입력 길이
    reranker_batch_size: int = 8  # 배치 크기
    reranker_num_workers: int = 0  # 데이터 로더 워커 수

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    max_retrieved_docs: int = 5  # 문서 수를 더 줄여 정확성 집중 (6 -> 5)
    similarity_threshold: float = 0.55  # 더욱 엄격한 임계값으로 품질 향상 (0.45 -> 0.55)
    max_context_length: int = 5000  # 컨텍스트 길이 단축 (6000 -> 5000)
    min_score_threshold: int = 8  # 품질 임계값 더욱 강화 (7점 -> 8점)
    max_retry_attempts: int = 2  # 재시도 횟수 줄여 효율성 향상
    
    # 신뢰도 향상을 위한 추가 설정
    keyword_weight: float = 0.4  # 키워드 매칭 가중치 더 증가 (0.35 -> 0.4)
    semantic_weight: float = 0.6  # 의미적 유사도 가중치 조정 (0.65 -> 0.6)
    diversity_threshold: int = 1  # 같은 소스에서 최대 문서 수 더 감소 (2 -> 1)
    min_text_length: int = 80  # 최소 텍스트 길이 증가 (50 -> 80)
    context_relevance_threshold: float = 0.8  # 컨텍스트 관련성 임계값 더욱 강화 (0.7 -> 0.8)
    
    # 새로운 품질 향상 설정
    intent_matching_weight: float = 0.4  # 의도 매칭 가중치 증가 (0.3 -> 0.4)
    sequence_matching_weight: float = 0.2  # 순서 매칭 가중치 유지
    important_keyword_boost: float = 2.5  # 중요 키워드 부스트 증가 (2.0 -> 2.5)
    combined_score_threshold: float = 0.6  # 종합 점수 임계값 강화 (0.5 -> 0.6)
    
    # Re-ranker 설정 (더 엄격하게)
    enable_reranker: bool = True  # Re-ranker 활성화
    reranker_top_k: int = 8  # Re-ranker에 입력할 상위 문서 수 감소 (10 -> 8)
    reranker_output_k: int = 5  # Re-ranker 출력 문서 수 감소 (6 -> 5)
    reranker_weight: float = 0.5  # BGE-Large Re-ranker 점수 가중치 증가 (0.4 -> 0.5)
    bm25_weight: float = 0.3  # BM25 점수 가중치 유지
    embedding_weight: float = 0.2  # 임베딩 점수 가중치 감소 (0.3 -> 0.2)
    diversity_penalty: float = 0.15  # MMR 다양성 패널티 증가 (0.1 -> 0.15)
    mmr_lambda: float = 0.7  # MMR 람다 파라미터 (관련성 vs 다양성 균형)

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