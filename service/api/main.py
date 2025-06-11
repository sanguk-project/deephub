from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
import uuid
from pathlib import Path

# RAG 시스템 컴포넌트 import
from service.core.rag_system import ask_with_context, get_rag_status, rag_query
from service.core.langgraph_rag import langgraph_ask_with_context, get_langgraph_rag_status
from service.core.composite_rag_system import composite_ask_with_context, get_composite_rag_status
from admin.tools.document_indexer import (
    index_document_file, 
    index_text, 
    get_indexer_status
)
from service.storage.vector_store import get_vector_store_info

# FastAPI 앱 생성
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
            "name": "System Status",
            "description": "시스템 상태 확인 엔드포인트",
        }
    ]
)

# 템플릿 설정
templates = Jinja2Templates(directory="service/web/template")

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="service/web/static"), name="static")

# Pydantic 모델 정의
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    confidence: float = 0.0
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
            confidence=rag_response.confidence_score,
            quality_score=round(rag_response.quality_score, 1),  # 소수 첫째자리까지만
            sources=sources,
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
            "answer": result["final_answer"],
            "confidence": result.get("confidence_score", 0.0),
            "sources": result.get("sources", []),
            "approved": result.get("approved", False),
            "review_summary": result.get("review_summary", ""),
            "processing_time": result.get("processing_time", 0.0),
            "retrieved_count": len(result.get("retrieved_documents", [])),
            "pipeline_metadata": result.get("pipeline_metadata", {}),
            "model_info": "복합 RAG: GPT-4.1 → EXAONE → GPT-4",
            "verification_score": result.get("pipeline_metadata", {}).get("gemini_verification_score", 0),
            "review_score": result.get("pipeline_metadata", {}).get("gpt4_review_score", 0),
            "error": result.get("error")
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
         description="5단계 복합 RAG 시스템(GPT-4.1 + EXAONE + GPT-4)의 상태를 조회합니다.")
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
                "step2": "GPT-4.1 정확도 검증",
                "step3": "EXAONE-3.5-2.4B-Instruct 답변 생성",
                "step4": "GPT-4 답변 검수",
                "step5": "최종 답변 확정"
            },
            "system_info": "5단계 복합 RAG 파이프라인"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"복합 RAG 상태 조회 중 오류: {str(e)}")

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
          summary="파일 업로드 및 즉시 인덱싱",
          description="파일을 업로드하고 즉시 벡터DB에 인덱싱합니다.")
async def upload_and_index_document(
    file: UploadFile = File(...),
    doc_title: str = Form(None)
):
    """파일 업로드 및 즉시 인덱싱"""
    try:
        # 지원하는 파일 확장자 확인
        supported_extensions = ['.pdf', '.txt', '.docx', '.md']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_extensions)}"
            )
        
        # 임시 파일 저장
        upload_id = str(uuid.uuid4())
        temp_filename = f"{upload_id}_{file.filename}"
        temp_file_path = Path("/tmp") / temp_filename
        
        # 파일 저장
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # 문서 디렉토리로 이동
        from shared.config.settings import settings
        documents_dir = Path(settings.path.documents_dir)
        documents_dir.mkdir(exist_ok=True)
        
        final_file_path = documents_dir / file.filename
        
        # 동일한 파일명이 있으면 번호 추가
        counter = 1
        while final_file_path.exists():
            name_without_ext = Path(file.filename).stem
            extension = Path(file.filename).suffix
            final_file_path = documents_dir / f"{name_without_ext}_{counter}{extension}"
            counter += 1
        
        # 최종 위치로 이동
        os.rename(temp_file_path, final_file_path)
        
        # 즉시 인덱싱
        from admin.tools.document_indexer import index_document_file
        
        metadata = {}
        if doc_title:
            metadata["title"] = doc_title
        metadata["upload_id"] = upload_id
        metadata["original_filename"] = file.filename
        
        success = index_document_file(str(final_file_path), metadata, force=True)
        
        if success:
            return {
                "success": True,
                "message": f"파일 '{file.filename}' 업로드 및 인덱싱 완료",
                "file_path": str(final_file_path),
                "upload_id": upload_id,
                "indexed": True
            }
        else:
            return {
                "success": False,
                "message": f"파일 업로드는 완료되었으나 인덱싱 실패: {file.filename}",
                "file_path": str(final_file_path),
                "upload_id": upload_id,
                "indexed": False
            }
        
    except Exception as e:
        # 임시 파일 정리
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"파일 업로드 및 인덱싱 중 오류: {str(e)}")

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
        success = index_text(text, doc_id, metadata)
        
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

# 개발 서버 실행용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9999) 