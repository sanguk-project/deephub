"""
문서 인덱서 - 실시간 문서 인덱싱 및 관리
LlamaIndex 통합 문서 로더를 사용한 다양한 형식 문서 처리
"""

import os
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# LlamaIndex 문서 처리 라이브러리
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.readers.file import (
    PDFReader,
    DocxReader, 
    UnstructuredReader,
    FlatReader
)
from llama_index.core import SimpleDirectoryReader

# 내부 모듈
from service.storage.vector_store import FAISSVectorStore, DocumentEmbedding
from shared.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class IndexingConfig:
    """문서 인덱싱 설정"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.txt', '.md', '.docx', '.doc', '.html', '.rtf']

@dataclass
class DocumentInfo:
    """문서 정보"""
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    last_modified: datetime
    format: str
    chunks_count: int = 0
    indexed_at: Optional[datetime] = None

class DocumentIndexer:
    """문서 인덱서 클래스 - LlamaIndex 기반"""
    
    def __init__(self, config: IndexingConfig = None):
        self.config = config or IndexingConfig()
        self.vector_store = FAISSVectorStore()
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.model.embedding_model
        )
        self.text_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.indexed_files: Dict[str, DocumentInfo] = {}
        
        # LlamaIndex 파일 확장자별 리더 매핑
        self.file_extractor = {
            ".pdf": PDFReader(),
            ".docx": DocxReader(),
            ".doc": DocxReader(),
            ".txt": FlatReader(),
            ".md": FlatReader(),
            ".html": UnstructuredReader(),
            ".rtf": UnstructuredReader()
        }
        
        logger.info("문서 인덱서 초기화 완료 (LlamaIndex 기반)")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"파일 해시 계산 실패 {file_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """LlamaIndex를 사용한 파일에서 텍스트 추출"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.file_extractor:
                logger.warning(f"지원하지 않는 파일 형식: {file_ext}")
                return ""
            
            # LlamaIndex 리더를 사용한 문서 로드
            reader = self.file_extractor[file_ext]
            documents = reader.load_data(file=Path(file_path))
            
            # 모든 문서의 텍스트 결합
            text_content = ""
            for doc in documents:
                text_content += doc.text + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"LlamaIndex 텍스트 추출 실패 {file_path}: {e}")
            return ""
    
    async def index_document(self, file_path: str, force_reindex: bool = False) -> bool:
        """단일 문서 인덱싱 - LlamaIndex 기반"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"파일이 존재하지 않음: {file_path}")
                return False
            
            file_path = os.path.abspath(file_path)
            file_name = os.path.basename(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            # 지원 형식 확인
            if file_ext not in self.config.supported_formats:
                logger.warning(f"지원하지 않는 파일 형식: {file_ext}")
                return False
            
            # 파일 정보 수집
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
            file_hash = self._calculate_file_hash(file_path)
            
            # 중복 인덱싱 확인
            if not force_reindex and file_path in self.indexed_files:
                existing_info = self.indexed_files[file_path]
                if existing_info.file_hash == file_hash:
                    logger.info(f"파일이 이미 인덱싱됨: {file_name}")
                    return True
            
            logger.info(f"문서 인덱싱 시작 (LlamaIndex): {file_name}")
            
            # LlamaIndex로 텍스트 추출
            text_content = self.extract_text_from_file(file_path)
            if not text_content:
                logger.warning(f"텍스트 추출 실패: {file_name}")
                return False
            
            # 텍스트 청킹
            text_chunks = self.text_splitter.split_text(text_content)
            if not text_chunks:
                logger.warning(f"청킹된 텍스트가 없음: {file_name}")
                return False
            
            # 임베딩 생성 및 벡터 DB 저장
            document_embeddings = []
            
            for i, chunk in enumerate(text_chunks):
                # 임베딩 생성
                embedding = self.embed_model.get_text_embedding(chunk)
                
                # 문서 임베딩 객체 생성
                doc_id = f"{file_hash}_{i}"
                doc_embedding = DocumentEmbedding(
                    id=doc_id,
                    text=chunk,
                    embedding=embedding,
                    metadata={
                        "source_file": file_name,
                        "file_path": file_path,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "file_size": file_size,
                        "last_modified": last_modified.isoformat(),
                        "file_hash": file_hash,
                        "format": file_ext,
                        "extraction_method": "llamaindex"
                    }
                )
                document_embeddings.append(doc_embedding)
            
            # 벡터 DB에 삽입
            success = await self.vector_store.insert_documents(document_embeddings)
            
            if success:
                # 인덱싱 정보 저장
                doc_info = DocumentInfo(
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                    file_hash=file_hash,
                    last_modified=last_modified,
                    format=file_ext,
                    chunks_count=len(text_chunks),
                    indexed_at=datetime.now()
                )
                self.indexed_files[file_path] = doc_info
                
                logger.info(f"문서 인덱싱 완료: {file_name} ({len(text_chunks)}개 청크)")
                return True
            else:
                logger.error(f"벡터 DB 삽입 실패: {file_name}")
                return False
                
        except Exception as e:
            logger.error(f"문서 인덱싱 중 오류 {file_path}: {e}")
            return False
    
    async def index_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """디렉토리 내 모든 문서 인덱싱 - LlamaIndex SimpleDirectoryReader 사용"""
        try:
            if not os.path.exists(directory_path):
                logger.error(f"디렉토리가 존재하지 않음: {directory_path}")
                return {"success": False, "error": "디렉토리가 존재하지 않음"}
            
            logger.info(f"디렉토리 인덱싱 시작 (LlamaIndex): {directory_path}")
            
            # LlamaIndex SimpleDirectoryReader 사용
            try:
                reader = SimpleDirectoryReader(
                    input_dir=directory_path,
                    recursive=recursive,
                    file_extractor=self.file_extractor,
                    required_exts=self.config.supported_formats
                )
                documents = reader.load_data()
                
                if not documents:
                    logger.warning(f"인덱싱할 문서가 없음: {directory_path}")
                    return {"success": True, "indexed_count": 0, "failed_count": 0}
                
                # 각 문서별로 개별 처리
                indexed_count = 0
                failed_count = 0
                
                for doc in documents:
                    try:
                        # 파일 경로에서 메타데이터 추출
                        file_path = doc.metadata.get('file_path', '')
                        if not file_path:
                            failed_count += 1
                            continue
                        
                        # 기존 index_document 메서드 재사용
                        success = await self.index_document(file_path)
                        if success:
                            indexed_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"개별 문서 처리 실패: {e}")
                        failed_count += 1
                
                logger.info(f"디렉토리 인덱싱 완료: {indexed_count}개 성공, {failed_count}개 실패")
                
                return {
                    "success": True,
                    "indexed_count": indexed_count,
                    "failed_count": failed_count,
                    "total_files": len(documents)
                }
                
            except Exception as e:
                logger.error(f"SimpleDirectoryReader 오류: {e}")
                # 폴백: 기존 방식으로 파일 수집
                return await self._fallback_directory_indexing(directory_path, recursive)
                
        except Exception as e:
            logger.error(f"디렉토리 인덱싱 중 오류 {directory_path}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_directory_indexing(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """폴백: 기존 방식의 디렉토리 인덱싱"""
        files_to_index = []
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in self.config.supported_formats:
                        files_to_index.append(file_path)
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in self.config.supported_formats:
                        files_to_index.append(file_path)
        
        if not files_to_index:
            return {"success": True, "indexed_count": 0, "failed_count": 0}
        
        # 순차 인덱싱
        results = []
        for file_path in files_to_index:
            result = await self.index_document(file_path)
            results.append(result)
        
        indexed_count = sum(results)
        failed_count = len(results) - indexed_count
        
        return {
            "success": True,
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "total_files": len(files_to_index)
        }
    
    async def index_text(self, text: str, source_name: str = "direct_input") -> bool:
        """텍스트 직접 인덱싱"""
        try:
            if not text.strip():
                logger.warning("빈 텍스트는 인덱싱할 수 없음")
                return False
            
            logger.info(f"텍스트 인덱싱 시작: {source_name}")
            
            # 텍스트 청킹
            text_chunks = self.text_splitter.split_text(text)
            if not text_chunks:
                logger.warning("청킹된 텍스트가 없음")
                return False
            
            # 임베딩 생성 및 벡터 DB 저장
            document_embeddings = []
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            for i, chunk in enumerate(text_chunks):
                # 임베딩 생성
                embedding = self.embed_model.get_text_embedding(chunk)
                
                # 문서 임베딩 객체 생성
                doc_id = f"{text_hash}_{i}"
                doc_embedding = DocumentEmbedding(
                    id=doc_id,
                    text=chunk,
                    embedding=embedding,
                    metadata={
                        "source_file": source_name,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "text_hash": text_hash,
                        "format": "text",
                        "extraction_method": "direct_input",
                        "indexed_at": datetime.now().isoformat()
                    }
                )
                document_embeddings.append(doc_embedding)
            
            # 벡터 DB에 삽입
            success = await self.vector_store.insert_documents(document_embeddings)
            
            if success:
                logger.info(f"텍스트 인덱싱 완료: {source_name} ({len(text_chunks)}개 청크)")
                return True
            else:
                logger.error(f"벡터 DB 삽입 실패: {source_name}")
                return False
                
        except Exception as e:
            logger.error(f"텍스트 인덱싱 중 오류: {e}")
            return False
    
    def get_indexer_status(self) -> Dict[str, Any]:
        """인덱서 상태 조회"""
        try:
            vector_info = self.vector_store.get_collection_info()
            
            return {
                "indexed_files_count": len(self.indexed_files),
                "total_documents": vector_info.get("total_entities", 0),
                "vector_dimension": vector_info.get("dimension", 0),
                "index_type": vector_info.get("index_type", "unknown"),
                "supported_formats": self.config.supported_formats,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "embedding_model": settings.model.embedding_model,
                "extraction_method": "llamaindex",
                "available_readers": list(self.file_extractor.keys())
            }
        except Exception as e:
            logger.error(f"인덱서 상태 조회 중 오류: {e}")
            return {"error": str(e)}

# 전역 인덱서 인스턴스
_document_indexer: Optional[DocumentIndexer] = None

def get_document_indexer() -> DocumentIndexer:
    """문서 인덱서 인스턴스 반환 (싱글톤)"""
    global _document_indexer
    if _document_indexer is None:
        _document_indexer = DocumentIndexer()
    return _document_indexer

# 편의 함수들
async def auto_index_documents():
    """설정된 문서 디렉토리 자동 인덱싱"""
    indexer = get_document_indexer()
    documents_dir = settings.path.documents_dir
    
    if os.path.exists(documents_dir):
        logger.info(f"자동 문서 인덱싱 시작 (LlamaIndex): {documents_dir}")
        result = await indexer.index_directory(documents_dir, recursive=True)
        logger.info(f"자동 인덱싱 결과: {result}")
        return result
    else:
        logger.warning(f"문서 디렉토리가 존재하지 않음: {documents_dir}")
        return {"success": False, "error": "문서 디렉토리 없음"}

async def index_document_file(file_path: str) -> bool:
    """파일 인덱싱 편의 함수"""
    indexer = get_document_indexer()
    return await indexer.index_document(file_path)

async def index_text(text: str, source_name: str = "api_input") -> bool:
    """텍스트 인덱싱 편의 함수"""
    indexer = get_document_indexer()
    return await indexer.index_text(text, source_name)

def get_indexer_status() -> Dict[str, Any]:
    """인덱서 상태 조회 편의 함수"""
    indexer = get_document_indexer()
    return indexer.get_indexer_status() 