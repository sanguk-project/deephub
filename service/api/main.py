from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
import multiprocessing

# RAG 시스템 컴포넌트 import
from service.core.rag_system import get_rag_status, rag_query
from service.core.langgraph_rag import langgraph_ask_with_context, get_langgraph_rag_status
from service.core.composite_rag_system import composite_ask_with_context, get_composite_rag_status, get_composite_rag_system
from admin.tools.document_indexer import (
    index_document_file, 
    index_text, 
    get_indexer_status,
)
from service.storage.vector_store import get_vector_store_info

# 로거 설정
logger = logging.getLogger(__name__)

# 멀티프로세싱 오류 방지 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱 생명주기 관리"""
    
    # 시작 시
    try:
        logger.info("RAG 시스템 초기화 중...")
        await get_composite_rag_system()
        logger.info("RAG 시스템 초기화 완료")
        yield
    finally:
        # 종료 시
        logger.info("RAG 시스템 리소스 정리 중...")
        
        # 멀티프로세싱 리소스 정리
        for p in multiprocessing.active_children():
            p.terminate()
            p.join(timeout=1)
        
        logger.info("RAG 시스템 리소스 정리 완료")

# FastAPI 앱 생성 (생명주기 관리 포함)
app = FastAPI(
    title="ILJoo Deep Hub", 
    description="RAG System with FAISS + LlamaIndex",
    version="2.0.0",
    tags_metadata=[
        {
            "name": "Frontend",
            "description": "웹 인터페이스 관련 엔드포인트",
        },
        {
            "name": "RAG Chat",
            "description": "RAG 기반 AI 챗봇 질의응답 엔드포인트",
        },
        {
            "name": "Document Indexing",
            "description": "백엔드 문서 인덱싱 관련 엔드포인트 (개발자용)",
        },
        {
            "name": "Document Upload",
            "description": "문서 업로드 및 실시간 인덱싱 엔드포인트",
        },
        {
            "name": "Memory Storage",
            "description": "휘발성 메모리 기반 문서 저장소 관리 엔드포인트",
        },
        {
            "name": "System Status",
            "description": "시스템 상태 확인 엔드포인트",
        },
        {
            "name": "VectorDB Management",
            "description": "VectorDB 적재 조건 관리 및 재적재 엔드포인트",
        }
    ],
    lifespan=lifespan
)

# 템플릿 설정
templates = Jinja2Templates(directory="service/web/template")

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="service/web/static"), name="static")

# === 메모리 기반 문서 저장 시스템 ===
class InMemoryDocumentStore:
    """메모리 기반 문서 저장소 (휘발성)"""
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        logger.info("메모리 기반 문서 저장소 초기화 완료")
    
    def store_document(self, doc_id: str, filename: str, content: bytes, metadata: Dict[str, Any] = None) -> bool:
        """문서를 메모리에 저장"""
        try:
            self.documents[doc_id] = {
                "filename": filename,
                "content": content,
                "metadata": metadata or {},
                "uploaded_at": datetime.now().isoformat(),
                "file_size": len(content)
            }
            logger.info(f"메모리에 문서 저장: {filename} (ID: {doc_id})")
            return True
        except Exception as e:
            logger.error(f"메모리 문서 저장 실패: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """메모리에서 문서 조회"""
        return self.documents.get(doc_id)
    
    def get_document_content(self, doc_id: str) -> bytes:
        """메모리에서 문서 내용 조회"""
        doc = self.documents.get(doc_id)
        return doc["content"] if doc else None
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """저장된 모든 문서 목록"""
        return [
            {
                "doc_id": doc_id,
                "filename": doc["filename"],
                "file_size": doc["file_size"],
                "uploaded_at": doc["uploaded_at"],
                "metadata": doc["metadata"]
            }
            for doc_id, doc in self.documents.items()
        ]
    
    def remove_document(self, doc_id: str) -> bool:
        """메모리에서 문서 제거"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.info(f"메모리에서 문서 제거: {doc_id}")
            return True
        return False
    
    def clear_all(self) -> int:
        """모든 문서 제거"""
        count = len(self.documents)
        self.documents.clear()
        logger.info(f"메모리에서 모든 문서 제거: {count}개")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        total_size = sum(doc["file_size"] for doc in self.documents.values())
        return {
            "total_documents": len(self.documents),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

# 전역 메모리 문서 저장소
memory_store = InMemoryDocumentStore()

# Pydantic 모델 정의
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = []
    quality_score: float = 0.0  # GPT-4.1 품질 평가 점수 (0-10)
    model_info: str = "기본 RAG + GPT-4.1 품질평가"

class TextIndexRequest(BaseModel):
    text: str
    doc_id: str = None
    metadata: Dict[str, Any] = {}

class FileIndexRequest(BaseModel):
    file_path: str
    metadata: Dict[str, Any] = {}

class IndexResponse(BaseModel):
    success: bool
    message: str
    indexed_count: int = 0

class SystemStatusResponse(BaseModel):
    rag_status: Dict[str, Any]
    indexer_status: Dict[str, Any]
    vector_store_info: Dict[str, Any]

@app.get("/favicon.ico", 
         tags=["Frontend"],
         summary="파비콘",
         description="브라우저 파비콘 요청에 대한 응답입니다.")
async def favicon():
    return FileResponse("service/web/static/favicon.ico")

@app.get("/", 
         response_class=HTMLResponse,
         tags=["Frontend"],
         summary="메인 페이지",
         description="일주 딥 허브 RAG 시스템 메인 페이지를 렌더링합니다.")
async def index(request: Request):
    initial_message = "일주 딥 허브입니다. 무엇을 도와드릴까요?"
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": initial_message
    })

@app.post("/ask", 
          response_model=AnswerResponse,
          tags=["RAG Chat"],
          summary="RAG 기반 질의응답 + GPT-4.1 품질평가",
          description="사용자의 질문을 받아서 기본 RAG 시스템을 통해 답변을 생성하고, GPT-4.1로 품질을 평가하여 MongoDB에 저장합니다.")
async def ask(question_data: QuestionRequest):
    question = question_data.question
    
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문이 제공되지 않았습니다.")
    
    try:
        # 업그레이드된 RAG 시스템을 통한 질의응답 (품질 평가 포함)
        rag_response = await rag_query(question)
        
        # 소스 파일 이름만 추출
        sources = [doc.get('source_file', 'unknown') for doc in rag_response.retrieved_documents]
        
        return AnswerResponse(
            answer=rag_response.answer,
            sources=sources,
            quality_score=round(rag_response.quality_score, 1),  # 소수 첫째자리까지만
            model_info=f"기본 RAG + GPT-4.1 품질평가 (점수: {rag_response.quality_score:.1f}/10)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의응답 처리 중 오류: {str(e)}")

@app.post("/ask-langgraph", 
          tags=["RAG Chat"],
          summary="LangGraph + GPT-4.1 검증 RAG 질의응답",
          description="LangGraph 워크플로우와 GPT-4.1을 사용한 검증된 답변을 생성합니다.")
async def ask_langgraph(question_data: QuestionRequest):
    question = question_data.question
    
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문이 제공되지 않았습니다.")
    
    try:
        # LangGraph RAG 시스템을 통한 질의응답
        result = await langgraph_ask_with_context(question)
        
        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
            "verification_score": result.get("verification_score", 0.0),
            "reasoning": result.get("reasoning", ""),
            "retrieved_count": result.get("retrieved_count", 0),
            "model_info": "LangGraph + GPT-4.1",
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph RAG 처리 중 오류: {str(e)}")

@app.post("/ask-composite", 
          tags=["RAG Chat"],
          summary="복합 RAG 파이프라인 질의응답",
          description="5단계 복합 RAG: 검색 → GPT-4.1 검증 → EXAONE 답변생성 → GPT-4 검수 → 최종답변")
async def ask_composite(question_data: QuestionRequest):
    question = question_data.question
    
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문이 제공되지 않았습니다.")
    
    try:
        # 복합 RAG 파이프라인 실행
        result = await composite_ask_with_context(question)
        
        return {
            "answer": result.get("final_answer", "답변을 생성할 수 없습니다."),
            "sources": result.get("sources", []),
            "quality_score": result.get("gpt_review_score", 0),
            "model_info": "Composite RAG (EXAONE + GPT-4.1 검수 + Re-ranker)",
            "processing_time": result.get("processing_time", 0),
            "retrieved_docs": len(result.get("retrieved_documents", [])),
            "pipeline_steps": 4
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"복합 RAG 파이프라인 처리 중 오류: {str(e)}")

# 개발자용 문서 인덱싱 엔드포인트들
@app.post("/admin/index-text",
          response_model=IndexResponse,
          tags=["Document Indexing"],
          summary="텍스트 인덱싱",
          description="개발자용: 텍스트를 직접 벡터 데이터베이스에 인덱싱합니다.")
async def admin_index_text(text_data: TextIndexRequest):
    """개발자용 텍스트 인덱싱 API"""
    try:
        success = index_text(
            text=text_data.text,
            doc_id=text_data.doc_id,
            metadata=text_data.metadata
        )
        
        if success:
            return IndexResponse(
                success=True,
                message="텍스트가 성공적으로 인덱싱되었습니다.",
                indexed_count=1
            )
        else:
            raise HTTPException(status_code=500, detail="텍스트 인덱싱에 실패했습니다.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인덱싱 중 오류: {str(e)}")

@app.post("/admin/index-file",
          response_model=IndexResponse,
          tags=["Document Indexing"],
          summary="파일 인덱싱",
          description="개발자용: 지정된 파일 경로의 문서를 인덱싱합니다.")
async def admin_index_file(file_data: FileIndexRequest):
    """개발자용 파일 인덱싱 API"""
    try:
        success = index_document_file(
            file_path=file_data.file_path,
            metadata=file_data.metadata
        )
        
        if success:
            return IndexResponse(
                success=True,
                message=f"파일 '{file_data.file_path}'이 성공적으로 인덱싱되었습니다.",
                indexed_count=1
            )
        else:
            raise HTTPException(status_code=500, detail="파일 인덱싱에 실패했습니다.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 인덱싱 중 오류: {str(e)}")

@app.get("/status",
         response_model=SystemStatusResponse,
         tags=["System Status"],
         summary="시스템 전체 상태",
         description="RAG 시스템, 인덱서, 벡터 스토어의 전체 상태를 조회합니다.")
async def get_system_status():
    """시스템 전체 상태 조회"""
    try:
        rag_status = await get_rag_status()
        indexer_status = get_indexer_status()
        vector_info = get_vector_store_info()
        
        return SystemStatusResponse(
            rag_status=rag_status,
            indexer_status=indexer_status,
            vector_store_info=vector_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 조회 중 오류: {str(e)}")

@app.get("/status-langgraph",
         tags=["System Status"],
         summary="LangGraph RAG 시스템 상태",
         description="LangGraph 기반 RAG 시스템과 GPT-4.1 검증 모델의 상태를 조회합니다.")
async def get_langgraph_system_status():
    """LangGraph RAG 시스템 상태 조회"""
    try:
        langgraph_status = await get_langgraph_rag_status()
        indexer_status = get_indexer_status()
        vector_info = get_vector_store_info()
        
        return {
            "langgraph_rag_status": langgraph_status,
            "indexer_status": indexer_status,
            "vector_store_info": vector_info,
            "system_info": "LangGraph + GPT-4.1 통합 시스템"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph 상태 조회 중 오류: {str(e)}")

@app.get("/status-composite",
         tags=["System Status"],
         summary="복합 RAG 시스템 상태",
         description="5단계 복합 RAG 시스템(GPT-4.1 + EXAONE + GPT-4 + Re-ranker)의 상태를 조회합니다.")
async def get_composite_system_status():
    """복합 RAG 시스템 상태 조회"""
    try:
        composite_status = await get_composite_rag_status()
        indexer_status = get_indexer_status()
        vector_info = get_vector_store_info()
        
        return {
            "composite_rag_status": composite_status,
            "indexer_status": indexer_status,
            "vector_store_info": vector_info,
            "status": "healthy" if composite_status.get("composite_rag_initialized", False) else "initializing",
            "message": "복합 RAG 시스템 상태 조회 완료",
            "models_status": composite_status.get("models_loaded", {}),
            "pipeline_info": {
                "step1": "RAG 문서 검색 (FAISS + BGE-M3)",
                "step1.5": "✨ Re-ranker (BGE-Large + BM25 + MMR)",
                "step2": "EXAONE-3.5-2.4B-Instruct 답변 생성", 
                "step3": "GPT-4.1 답변 검수",
                "step4": "최종 답변 확정"
            },
            "reranker_info": {
                "enabled": composite_status.get("config", {}).get("enable_reranker", False),
                "algorithm": "Hybrid BGE-Large + BM25 + Embedding",
                "diversity": "MMR (Maximal Marginal Relevance)"
            },
            "system_info": "4단계 복합 RAG 파이프라인 + Re-ranker"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"복합 RAG 상태 조회 중 오류: {str(e)}")

@app.get("/status-reranker",
         tags=["System Status"],
         summary="Re-ranker 시스템 상태", 
         description="하이브리드 Re-ranker 시스템(BGE-Large + BM25 + MMR)의 상태를 조회합니다.")
async def get_reranker_status():
    """Re-ranker 시스템 상태 조회 (settings.py 설정 정보 포함)"""
    try:
        from shared.config.settings import settings
        
        # 복합 RAG 시스템 인스턴스 가져오기
        composite_system = await get_composite_rag_system()
        
        if composite_system.reranker is None:
            return {
                "status": "disabled" if not settings.rag.enable_reranker else "error",
                "message": "Re-ranker가 비활성화되어 있거나 로드 실패",
                "settings_config": {
                    "enable_reranker": settings.rag.enable_reranker,
                    "model_name": settings.model.reranker_model,
                    "model_type": settings.model.reranker_model_type,
                    "device": settings.model.reranker_device
                },
                "suggestion": "settings.py에서 enable_reranker=True로 설정하고 모델이 정상 로드되는지 확인하세요"
            }
        
        # Re-ranker 상태 확인
        reranker_status = composite_system.reranker.get_reranker_status()
        system_status = composite_system.reranker.get_system_status()
        
        return {
            "status": "active",
            "message": "Re-ranker 시스템이 정상 작동중입니다.",
            "reranker_details": reranker_status,
            "system_details": system_status,
            "model_config": {
                "model_name": settings.model.reranker_model,
                "model_type": settings.model.reranker_model_type,
                "device": settings.model.reranker_device,
                "max_length": settings.model.reranker_max_length,
                "batch_size": settings.model.reranker_batch_size
            },
            "algorithm_config": {
                "type": "하이브리드 Re-ranker",
                "components": ["BGE-Large", "BM25", "임베딩", "MMR"],
                "input_docs": settings.rag.reranker_top_k,
                "output_docs": settings.rag.reranker_output_k,
                "score_weights": {
                    "reranker": settings.rag.reranker_weight,
                    "bm25": settings.rag.bm25_weight,
                    "embedding": settings.rag.embedding_weight
                },
                "mmr_config": {
                    "lambda": settings.rag.mmr_lambda,
                    "diversity_penalty": settings.rag.diversity_penalty
                }
            },
            "performance_metrics": {
                "expected_improvement": {
                    "accuracy": "75% → 90%",
                    "relevance": "65% → 85%",
                    "response_time": "3초 → 2초"
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Re-ranker 상태 조회 중 오류: {str(e)}",
            "error_details": str(e),
            "settings_config": {
                "enable_reranker": settings.rag.enable_reranker,
                "model_name": settings.model.reranker_model
            },
            "suggestion": "로그를 확인하고 필요한 패키지가 설치되어 있는지 확인하세요"
        }

@app.get("/health",
         tags=["System Status"],
         summary="서버 상태 확인",
         description="서버의 기본 상태를 확인합니다.")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "message": "ILJoo Deep Hub RAG System (FAISS + LlamaIndex) is running",
        "version": "2.0.0"
    }

# === 로깅 및 분석 엔드포인트 (MongoDB 제거됨) ===

# === 실시간 문서 인덱싱 기능만 유지 ===

@app.post("/admin/upload-document",
          tags=["Document Upload"],
          summary="파일 업로드 및 즉시 인덱싱 (메모리 기반)",
          description="파일을 메모리에 저장하고 즉시 벡터DB에 인덱싱합니다. (휘발성 - 프로그램 종료시 삭제)")
async def upload_and_index_document(
    file: UploadFile = File(...)
):
    """파일 업로드 및 즉시 인덱싱 (메모리 기반 - 휘발성)"""
    try:
        # 지원하는 파일 확장자 확인
        supported_extensions = ['.pdf', '.txt', '.docx', '.md']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_extensions)}"
            )
        
        # 고유 문서 ID 생성
        doc_id = str(uuid.uuid4())
        
        # 파일 내용을 메모리로 읽기
        content = await file.read()
        
        # 메모리 저장소에 문서 저장
        metadata = {
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_extension": file_extension,
            "upload_method": "web_interface"
        }
        
        store_success = memory_store.store_document(
            doc_id=doc_id,
            filename=file.filename,
            content=content,
            metadata=metadata
        )
        
        if not store_success:
            raise HTTPException(status_code=500, detail="메모리 저장 실패")
        
        # 즉시 인덱싱 (메모리 기반)
        from admin.tools.document_indexer import index_memory_document
        
        index_success = await index_memory_document(doc_id, file.filename, content, metadata)
        
        if index_success:
            # 저장소 통계
            stats = memory_store.get_stats()
            
            return {
                "success": True,
                "message": f"파일 '{file.filename}' 메모리 업로드 및 인덱싱 완료 (휘발성)",
                "doc_id": doc_id,
                "filename": file.filename,
                "file_size": len(content),
                "storage_type": "memory",
                "indexed": True,
                "memory_stats": stats
            }
        else:
            # 인덱싱 실패 시 메모리에서도 제거
            memory_store.remove_document(doc_id)
            return {
                "success": False,
                "message": f"파일 메모리 저장은 완료되었으나 인덱싱 실패: {file.filename}",
                "doc_id": doc_id,
                "indexed": False
            }
        
    except Exception as e:
        import traceback
        logger.error(f"메모리 기반 파일 업로드 및 인덱싱 중 오류: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=f"메모리 기반 파일 업로드 및 인덱싱 중 오류: {str(e)}")

@app.post("/admin/quick-text",
          tags=["Document Upload"],
          summary="텍스트 즉시 인덱싱",
          description="텍스트를 즉시 벡터DB에 인덱싱합니다.")
async def quick_text_index(
    text: str = Form(...),
    title: str = Form("사용자 입력 텍스트"),
    category: str = Form("general")
):
    """텍스트 즉시 인덱싱"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
        
        # 고유 문서 ID 생성
        doc_id = f"quick_text_{uuid.uuid4()}"
        
        # 메타데이터 구성
        metadata = {
            "title": title,
            "category": category,
            "type": "quick_text",
            "doc_id": doc_id
        }
        
        # 즉시 인덱싱
        from admin.tools.document_indexer import index_text
        success = await index_text(text, doc_id)
        
        if success:
            return {
                "success": True,
                "message": f"텍스트 '{title}' 인덱싱 완료",
                "doc_id": doc_id,
                "text_length": len(text),
                "indexed": True
            }
        else:
            return {
                "success": False,
                "message": f"텍스트 인덱싱 실패: {title}",
                "doc_id": doc_id,
                "indexed": False
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 인덱싱 중 오류: {str(e)}")

@app.get("/admin/indexing-status",
         tags=["Document Upload"],
         summary="인덱싱 상태 조회",
         description="현재 벡터DB 인덱싱 상태를 조회합니다.")
async def get_indexing_status():
    """인덱싱 상태 조회"""
    try:
        from admin.tools.document_indexer import get_indexer_status
        from service.storage.vector_store import get_vector_store_info
        
        indexer_status = get_indexer_status()
        vector_info = get_vector_store_info()
        
        return {
            "success": True,
            "indexer_status": indexer_status,
            "vector_store_info": vector_info,
            "total_documents": vector_info.get("total_entities", 0),
            "documents_directory": indexer_status.get("documents_directory"),
            "files_in_directory": indexer_status.get("files_in_directory", 0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "indexer_status": {},
            "vector_store_info": {}
        }

# === 메모리 저장소 관리 엔드포인트 ===

@app.get("/admin/memory-documents",
         tags=["Memory Storage"],
         summary="메모리 문서 목록 조회",
         description="현재 메모리에 저장된 모든 문서 목록을 조회합니다.")
async def get_memory_documents():
    """메모리 저장된 문서 목록 조회"""
    try:
        documents = memory_store.list_documents()
        stats = memory_store.get_stats()
        
        return {
            "success": True,
            "documents": documents,
            "stats": stats,
            "message": f"총 {len(documents)}개 문서가 메모리에 저장됨"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "documents": [],
            "stats": {}
        }

@app.get("/admin/memory-document/{doc_id}",
         tags=["Memory Storage"],
         summary="특정 메모리 문서 조회",
         description="문서 ID로 특정 메모리 문서의 상세 정보를 조회합니다.")
async def get_memory_document(doc_id: str):
    """특정 메모리 문서 조회"""
    try:
        document = memory_store.get_document(doc_id)
        
        if document:
            # content는 너무 크므로 제외하고 메타데이터만 반환
            return {
                "success": True,
                "doc_id": doc_id,
                "filename": document["filename"],
                "file_size": document["file_size"],
                "uploaded_at": document["uploaded_at"],
                "metadata": document["metadata"]
            }
        else:
            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없음: {doc_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/admin/memory-document/{doc_id}",
           tags=["Memory Storage"],
           summary="메모리 문서 삭제",
           description="특정 문서를 메모리에서 삭제합니다. (벡터DB에서는 유지됨)")
async def delete_memory_document(doc_id: str):
    """메모리에서 문서 삭제"""
    try:
        success = memory_store.remove_document(doc_id)
        
        if success:
            stats = memory_store.get_stats()
            return {
                "success": True,
                "message": f"문서 {doc_id}가 메모리에서 삭제됨",
                "remaining_stats": stats
            }
        else:
            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없음: {doc_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/admin/memory-documents/clear",
           tags=["Memory Storage"],
           summary="모든 메모리 문서 삭제",
           description="메모리에 저장된 모든 문서를 삭제합니다. ⚠️ 복구 불가능!")
async def clear_all_memory_documents():
    """모든 메모리 문서 삭제"""
    try:
        count = memory_store.clear_all()
        
        return {
            "success": True,
            "message": f"모든 메모리 문서 삭제 완료: {count}개 문서",
            "cleared_count": count
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cleared_count": 0
        }

@app.get("/admin/memory-stats",
         tags=["Memory Storage"],
         summary="메모리 저장소 통계",
         description="메모리 저장소의 사용량 및 통계 정보를 조회합니다.")
async def get_memory_stats():
    """메모리 저장소 통계"""
    try:
        stats = memory_store.get_stats()
        documents = memory_store.list_documents()
        
        # 파일 유형별 통계
        type_stats = {}
        for doc in documents:
            ext = doc["metadata"].get("file_extension", "unknown")
            if ext not in type_stats:
                type_stats[ext] = {"count": 0, "total_size": 0}
            type_stats[ext]["count"] += 1
            type_stats[ext]["total_size"] += doc["file_size"]
        
        return {
            "success": True,
            "overall_stats": stats,
            "type_breakdown": type_stats,
            "storage_info": {
                "storage_type": "memory",
                "volatile": True,
                "description": "프로그램 종료 시 모든 데이터 삭제됨"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# === VectorDB 관리 엔드포인트 ===

@app.get("/admin/vector-config-status",
         tags=["VectorDB Management"],
         summary="VectorDB 설정 상태 확인",
         description="VectorDB 적재 조건 변경 여부 및 현재 설정 상태를 확인합니다.")
async def get_vector_config_status():
    """VectorDB 설정 상태 확인"""
    try:
        from service.storage.vector_config_manager import get_config_status
        
        status = get_config_status()
        return {
            "success": True,
            "config_changed": status["config_changed"],
            "current_config": status["current_config"],
            "saved_config": status["saved_config"],
            "config_file_exists": status["config_file_exists"],
            "documents_dir_exists": status["documents_dir_exists"],
            "faiss_index_dir": status["faiss_index_dir"],
            "message": "설정 변경 감지됨" if status["config_changed"] else "설정 변경 없음"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VectorDB 설정 상태 확인 중 오류: {str(e)}")

@app.post("/admin/vector-auto-reindex",
          tags=["VectorDB Management"],
          summary="자동 설정 확인 및 재적재",
          description="VectorDB 적재 조건 변경을 확인하고 필요시 자동으로 초기화 및 재적재를 수행합니다.")
async def vector_auto_reindex():
    """자동 설정 확인 및 재적재"""
    try:
        from service.storage.vector_config_manager import auto_check_and_update
        
        updated = await auto_check_and_update()
        
        return {
            "success": True,
            "updated": updated,
            "message": "설정 변경으로 인한 자동 재적재 완료" if updated else "설정 변경 없음 - 재적재 불필요"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"자동 재적재 중 오류: {str(e)}")

@app.post("/admin/vector-force-reindex",
          tags=["VectorDB Management"],
          summary="강제 초기화 및 재적재",
          description="VectorDB를 강제로 초기화하고 모든 문서를 재적재합니다. ⚠️ 기존 벡터 데이터가 모두 삭제됩니다!")
async def vector_force_reindex():
    """강제 VectorDB 초기화 및 재적재"""
    try:
        from service.storage.vector_config_manager import force_reset_and_reindex
        
        await force_reset_and_reindex()
        
        return {
            "success": True,
            "message": "VectorDB 강제 초기화 및 재적재 완료"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"강제 재적재 중 오류: {str(e)}")

@app.get("/admin/vector-config-info",
         tags=["VectorDB Management"],
         summary="현재 VectorDB 설정 정보",
         description="현재 적용중인 VectorDB 적재 조건들을 상세히 조회합니다.")
async def get_vector_config_info():
    """현재 VectorDB 설정 정보 조회"""
    try:
        from service.storage.vector_config_manager import get_vector_config_manager
        
        manager = get_vector_config_manager()
        config_info = manager.get_current_config_info()
        
        return {
            "success": True,
            "config": config_info,
            "message": "VectorDB 설정 정보 조회 완료"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 정보 조회 중 오류: {str(e)}")

# 개발 서버 실행용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9999) 