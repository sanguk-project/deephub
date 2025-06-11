"""
FAISS 벡터 스토어 컴포넌트 (CPU 전용)
문서 임베딩 저장 및 검색을 담당
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# FAISS 로깅 최적화
logging.getLogger('faiss._loader').setLevel(logging.CRITICAL)
logging.getLogger('faiss.loader').setLevel(logging.CRITICAL)

import faiss
from pydantic import BaseModel

# FAISS 로거 레벨 조정
faiss_logger = logging.getLogger("faiss")
faiss_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

@dataclass
class FAISSConfig:
    """FAISS CPU 전용 설정"""
    def __init__(self):
        from shared.config.settings import settings
        self.index_dir = settings.path.faiss_index_dir
        self.index_file = "vector_index.faiss"
        self.metadata_file = "metadata.pkl"
        self.dimension = 1024  # BAAI/bge-m3 dimension
        self.index_type = "IndexIVFFlat"  # IVF (Inverted File) for better search performance
        self.nlist = 256  # IVF 클러스터 수 (인덱스 크기에 따라 조정)
        self.nprobe = 32  # 검색 시 탐색할 클러스터 수
        self.num_threads = 8  # CPU 스레드 수
        self.train_threshold = 256  # IVF 훈련을 위한 최소 벡터 수

class DocumentEmbedding(BaseModel):
    """문서 임베딩 데이터 모델"""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

class FAISSVectorStore:
    """FAISS 벡터 스토어 클래스 (CPU 전용)"""
    
    def __init__(self, config: FAISSConfig = None):
        self.config = config or FAISSConfig()
        self.index: Optional[faiss.Index] = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        
        # CPU 스레드 설정
        faiss.omp_set_num_threads(self.config.num_threads)
        
        # 인덱스 디렉토리 생성
        os.makedirs(self.config.index_dir, exist_ok=True)
        
        self._setup_index()
        self._load_existing_data()
    
    def _setup_index(self):
        """FAISS CPU 인덱스 설정"""
        try:
            # CPU 전용 인덱스 생성
            if self.config.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.config.dimension)
                logger.info("IndexFlatIP 생성 (코사인 유사도)")
            elif self.config.index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(self.config.dimension)
                logger.info("IndexFlatL2 생성 (유클리드 거리)")
            elif self.config.index_type == "IndexIVFFlat":
                # IVF (Inverted File) 인덱스 생성
                quantizer = faiss.IndexFlatIP(self.config.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.config.dimension, self.config.nlist)
                self.index.nprobe = self.config.nprobe
                logger.info(f"IndexIVFFlat 생성 (nlist={self.config.nlist}, nprobe={self.config.nprobe})")
            else:
                # 기본값으로 IVF 사용
                quantizer = faiss.IndexFlatIP(self.config.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.config.dimension, self.config.nlist)
                self.index.nprobe = self.config.nprobe
                logger.info(f"기본 IndexIVFFlat 생성")
            
            logger.info(f"FAISS CPU 인덱스 초기화 완료: {self.config.index_type} (CPU 스레드: {self.config.num_threads})")
                
        except Exception as e:
            logger.error(f"FAISS CPU 인덱스 설정 실패: {e}")
            raise e
    
    def _load_existing_data(self):
        """기존 인덱스와 메타데이터 로드"""
        index_path = os.path.join(self.config.index_dir, self.config.index_file)
        metadata_path = os.path.join(self.config.index_dir, self.config.metadata_file)
        
        # 인덱스 파일 로드
        if os.path.exists(index_path):
            try:
                # CPU 인덱스 직접 로드
                self.index = faiss.read_index(index_path)
                logger.info(f"기존 FAISS CPU 인덱스 로드: {index_path}")
            except Exception as e:
                logger.warning(f"인덱스 로드 실패, 새로 생성: {e}")
                self._setup_index()
        
        # 메타데이터 로드
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get('metadata_store', {})
                    self.id_to_index = data.get('id_to_index', {})
                    self.index_to_id = data.get('index_to_id', {})
                    self.next_index = data.get('next_index', 0)
                logger.info(f"기존 메타데이터 로드: {len(self.metadata_store)}개 문서")
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {e}")
    
    def _save_data(self):
        """인덱스와 메타데이터 저장"""
        try:
            # CPU 인덱스 직접 저장
            index_path = os.path.join(self.config.index_dir, self.config.index_file)
            faiss.write_index(self.index, index_path)
            
            # 메타데이터 저장
            metadata_path = os.path.join(self.config.index_dir, self.config.metadata_file)
            with open(metadata_path, 'wb') as f:
                data = {
                    'metadata_store': self.metadata_store,
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'next_index': self.next_index
                }
                pickle.dump(data, f)
            
            logger.info("FAISS CPU 인덱스 및 메타데이터 저장 완료")
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {e}")
    
    def _normalize_vector(self, vector: List[float]) -> np.ndarray:
        """벡터를 정규화 (cosine similarity를 위해)"""
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    async def insert_documents(self, documents: List[DocumentEmbedding]) -> bool:
        """문서 임베딩을 FAISS에 삽입"""
        try:
            vectors = []
            new_metadata = {}
            new_id_mapping = {}
            
            for doc in documents:
                # 이미 존재하는 문서 ID 확인
                if doc.id in self.id_to_index:
                    logger.warning(f"문서 ID가 이미 존재함: {doc.id}")
                    continue
                
                # 벡터 정규화
                normalized_vector = self._normalize_vector(doc.embedding)
                vectors.append(normalized_vector)
                
                # 메타데이터 저장
                current_index = self.next_index
                new_metadata[current_index] = {
                    "id": doc.id,
                    "text": doc.text,
                    "source_file": doc.metadata.get("source_file", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    **doc.metadata
                }
                
                # ID 매핑 저장
                new_id_mapping[doc.id] = current_index
                self.next_index += 1
            
            if not vectors:
                logger.warning("삽입할 문서가 없습니다")
                return True
            
            # FAISS 인덱스에 벡터 추가
            vectors_array = np.vstack(vectors)
            
            # IVF 인덱스인 경우 훈련 처리
            if isinstance(self.index, faiss.IndexIVFFlat):
                if not self.index.is_trained:
                    # 첫 번째 배치: 인덱스 훈련 필요
                    if len(vectors) >= self.config.train_threshold:
                        logger.info(f"IVF 인덱스 훈련 시작 (훈련 데이터: {len(vectors)}개)")
                        self.index.train(vectors_array)
                        logger.info("IVF 인덱스 훈련 완료")
                    else:
                        # 훈련 데이터가 부족한 경우, 기존 데이터와 합쳐서 훈련
                        all_vectors = []
                        if self.metadata_store:
                            # 기존 메타데이터에서 벡터 복원 (비추천 - 성능상)
                            logger.warning(f"훈련 데이터 부족 ({len(vectors)}개 < {self.config.train_threshold}개)")
                        
                        if len(vectors) >= 32:  # 최소한의 훈련 데이터
                            logger.info(f"최소 데이터로 IVF 인덱스 훈련 (데이터: {len(vectors)}개)")
                            self.index.train(vectors_array)
                        else:
                            # 너무 적은 데이터는 Flat 인덱스로 대체
                            logger.warning("데이터가 너무 적어 IndexFlatIP로 임시 변경")
                            self.index = faiss.IndexFlatIP(self.config.dimension)
                
                # 동적 nprobe 조정 (데이터 크기에 따라)
                current_total = self.index.ntotal + len(vectors)
                if current_total > 10000:
                    self.index.nprobe = min(64, self.config.nlist // 4)
                elif current_total > 1000:
                    self.index.nprobe = min(32, self.config.nlist // 8)
                else:
                    self.index.nprobe = min(16, self.config.nlist // 16)
                
                logger.debug(f"nprobe 조정: {self.index.nprobe} (총 데이터: {current_total}개)")
            
            self.index.add(vectors_array)
            
            # 메타데이터 업데이트
            self.metadata_store.update(new_metadata)
            self.id_to_index.update(new_id_mapping)
            
            # 역방향 매핑 업데이트
            for doc_id, idx in new_id_mapping.items():
                self.index_to_id[idx] = doc_id
            
            # 데이터 저장
            self._save_data()
            
            logger.info(f"{len(documents)}개 문서 FAISS CPU 인덱스에 삽입 완료")
            return True
            
        except Exception as e:
            logger.error(f"문서 삽입 중 오류: {e}")
            return False
    
    async def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """유사 문서 검색"""
        try:
            if self.index.ntotal == 0:
                logger.warning("인덱스가 비어있습니다")
                return []
            
            # 쿼리 벡터 정규화
            query_vector = self._normalize_vector(query_embedding)
            query_vector = query_vector.reshape(1, -1)
            
            # 검색 수행
            scores, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
            
            # 결과 포맷팅
            similar_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS에서 -1은 유효하지 않은 인덱스
                    continue
                    
                if idx in self.metadata_store:
                    metadata = self.metadata_store[idx]
                    similar_docs.append({
                        "id": metadata["id"],
                        "text": metadata["text"],
                        "score": float(score),
                        "source_file": metadata.get("source_file", ""),
                        "chunk_index": metadata.get("chunk_index", 0)
                    })
            
            logger.debug(f"검색 완료: {len(similar_docs)}개 결과")
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"유사 문서 검색 중 오류: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            return {
                "collection_name": "faiss_documents",
                "total_entities": self.index.ntotal if self.index else 0,
                "dimension": self.config.dimension,
                "index_type": self.config.index_type,
                "device": "CPU",
                "index_directory": self.config.index_dir
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 중 오류: {e}")
            return {}
    
    def delete_collection(self):
        """컬렉션 삭제 (개발/테스트용)"""
        try:
            # 인덱스 재설정
            self._setup_index()
            
            # 메타데이터 초기화
            self.metadata_store.clear()
            self.id_to_index.clear()
            self.index_to_id.clear()
            self.next_index = 0
            
            # 파일 삭제
            index_path = os.path.join(self.config.index_dir, self.config.index_file)
            metadata_path = os.path.join(self.config.index_dir, self.config.metadata_file)
            
            for path in [index_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info("FAISS 컬렉션 삭제 완료")
            
        except Exception as e:
            logger.error(f"컬렉션 삭제 중 오류: {e}")

# 글로벌 인스턴스
faiss_store = FAISSVectorStore()

# 편의 함수들
async def insert_document_embeddings(documents: List[DocumentEmbedding]) -> bool:
    """문서 임베딩 삽입 편의 함수"""
    return await faiss_store.insert_documents(documents)

async def search_similar_documents(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """유사 문서 검색 편의 함수"""
    return await faiss_store.search_similar(query_embedding, limit)

def get_vector_store_info() -> Dict[str, Any]:
    """벡터 스토어 정보 조회 편의 함수"""
    return faiss_store.get_collection_info() 