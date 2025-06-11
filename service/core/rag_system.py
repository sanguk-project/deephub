"""
RAG 시스템 통합 컴포넌트
검색 증강 생성(Retrieval-Augmented Generation) 기능을 제공
GPT-4.1 기반 품질 평가 및 MongoDB 로깅 포함
"""

import logging
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from service.storage.vector_store import search_similar_documents
from admin.tools.document_indexer import get_document_indexer
from service.storage.rag_logger import evaluate_and_log_rag, quick_rag_score, RAGEvaluationResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    embedding_model: str = "BAAI/bge-m3"
    max_retrieved_docs: int = 5
    similarity_threshold: float = 0.3
    max_context_length: int = 4000

class RAGResponse(BaseModel):
    """RAG 응답 모델"""
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    context_used: str
    confidence_score: float
    quality_score: float = 0.0  # GPT-4.1 품질 평가 점수 (0-10)
    evaluation_result: Dict[str, Any] = None  # 상세 평가 결과

class RAGSystem:
    """RAG 시스템 메인 클래스"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # 임베딩 모델 초기화
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model
        )
        
        logger.info("RAG 시스템 초기화 완료")
    
    async def query(self, question: str, max_docs: int = None) -> RAGResponse:
        """
        RAG 쿼리 실행
        
        Args:
            question (str): 사용자 질문
            max_docs (int): 최대 검색 문서 수
            
        Returns:
            RAGResponse: RAG 응답
        """
        try:
            max_docs = max_docs or self.config.max_retrieved_docs
            
            # 1. 질문 임베딩 생성
            question_embedding = self.embed_model.get_text_embedding(question)
            
            # 2. 유사 문서 검색 (더 많이 검색해서 중복 제거 후 필터링)
            similar_docs = await search_similar_documents(
                query_embedding=question_embedding,
                limit=max_docs * 3  # 중복 제거를 위해 더 많이 검색
            )
            
            # 3. 같은 source_file 중복 제거 및 통합
            deduplicated_docs = self._deduplicate_by_source_file(similar_docs)
            
            # 4. 검색 결과 필터링 (유사도 임계값)
            filtered_docs = [
                doc for doc in deduplicated_docs 
                if doc.get('score', 0) >= self.config.similarity_threshold
            ]
            
            # 5. 요청된 문서 수로 제한
            filtered_docs = filtered_docs[:max_docs]
            
            # 6. 컨텍스트 생성
            context = self._build_context(filtered_docs)
            
            # 7. 답변 생성 (현재는 기본 응답, 향후 LLM 연동)
            answer = self._generate_answer(question, context, filtered_docs)
            
            # 8. 신뢰도 점수 계산
            confidence = self._calculate_confidence(filtered_docs)
            
            # 9. RAG 품질 평가 및 MongoDB 로깅 (백그라운드에서 실행)
            quality_score = 0.0
            evaluation_result = None
            
            try:
                # 백그라운드 태스크로 품질 평가 및 로깅 실행
                if context and answer:
                    asyncio.create_task(
                        self._evaluate_and_log_quality(question, context, answer)
                    )
                    
                    # 빠른 품질 점수만 가져오기 (블로킹하지 않음)
                    quality_score = await quick_rag_score(question, context, answer)
                    
            except Exception as e:
                logger.warning(f"RAG 품질 평가 중 오류 (무시됨): {e}")
            
            return RAGResponse(
                answer=answer,
                retrieved_documents=filtered_docs,
                context_used=context,
                confidence_score=confidence,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"RAG 쿼리 실행 중 오류: {e}")
            return RAGResponse(
                answer="죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.",
                retrieved_documents=[],
                context_used="",
                confidence_score=0.0
            )
    
    def _deduplicate_by_source_file(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """같은 source_file의 문서들을 중복 제거하고 통합"""
        if not documents:
            return []
        
        # source_file별로 그룹화
        file_groups = {}
        for doc in documents:
            source_file = doc.get('source_file', 'unknown')
            if source_file not in file_groups:
                file_groups[source_file] = []
            file_groups[source_file].append(doc)
        
        # 각 파일별로 최고 점수 청크를 대표로 선택하고 텍스트 통합
        deduplicated = []
        for source_file, docs in file_groups.items():
            # 점수순으로 정렬
            docs.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # 최고 점수 문서를 기준으로 텍스트 통합
            representative_doc = docs[0].copy()
            
            # 상위 5개 청크의 텍스트를 결합 (더 많은 정보 포함)
            combined_texts = []
            for doc in docs[:5]:
                text = doc.get('text', '').strip()
                if text and text not in combined_texts:
                    combined_texts.append(text)
            
            # 텍스트 결합
            combined_text = '\n\n'.join(combined_texts)
            representative_doc['text'] = combined_text
            representative_doc['combined_chunks'] = len(docs)
            
            deduplicated.append(representative_doc)
        
        # 점수순으로 정렬하여 반환
        deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
        return deduplicated
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """검색된 문서들로부터 컨텍스트 구성"""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = doc.get('text', '')
            source_file = doc.get('source_file', 'unknown')
            
            # 컨텍스트 길이 제한 확인
            if current_length + len(doc_text) > self.config.max_context_length:
                break
            
            context_part = f"[문서 {i+1} - {source_file}]\n{doc_text}\n"
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, documents: List[Dict[str, Any]]) -> str:
        """
        답변 생성 (간결한 버전)
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 컨텍스트
            documents (List[Dict[str, Any]]): 검색된 문서들
            
        Returns:
            str: 생성된 답변
        """
        if not documents:
            return "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다. 문서가 인덱싱되어 있는지 확인해주세요."
        
        # 컨텍스트에서 핵심 정보만 추출
        context_lines = context.split('\n')
        answer_lines = []
        
        for line in context_lines:
            line = line.strip()
            # 문서 헤더 라인은 제외
            if line.startswith('[문서') or not line:
                continue
            # 핵심 내용만 포함
            answer_lines.append(line)
        
        # 질문에서 키워드 추출 (간단한 버전)
        question_lower = question.lower()
        question_keywords = []
        
        # 질문 유형별 키워드 매핑
        keyword_patterns = {
            '위치': ['위치', '주소', '부서', '층', '건물', '곳'],
            '시간': ['시간', '언제', '몇시', '시각', '일정'],
            '방법': ['방법', '어떻게', '절차', '과정', '단계'],
            '연락처': ['연락처', '전화', '번호', '메일', '이메일'],
            '차량': ['차량', '자동차', '셀토스', 'xm3', '주차', '예약'],
            '비용': ['비용', '요금', '가격', '돈', '비', '금액'],
            '사용': ['사용', '이용', '활용', '접근', '로그인']
        }
        
        # 질문에서 관련 키워드 찾기
        for category, keywords in keyword_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                question_keywords.extend(keywords)
        
        # 답변 조합
        if answer_lines:
            # 키워드가 있으면 관련성 있는 내용만 필터링
            if question_keywords:
                filtered_content = []
                for line in answer_lines:
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in question_keywords):
                        filtered_content.append(line)
                
                if filtered_content:
                    return '\n'.join(filtered_content)
            
            # 키워드 매칭이 없거나 결과가 없으면 관련도 높은 내용 반환
            # 긴 줄이나 의미있는 내용 우선
            meaningful_lines = []
            for line in answer_lines:
                # 너무 짧거나 의미없는 줄 제외
                if len(line.strip()) > 10 and not line.strip().startswith(('-', '•', '*')):
                    meaningful_lines.append(line)
            
            if meaningful_lines:
                return '\n'.join(meaningful_lines[:3])  # 최대 3줄
            else:
                return '\n'.join(answer_lines[:5])  # 최대 5줄
        else:
            return "관련 정보를 찾았지만 적절한 답변을 추출할 수 없습니다."
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """신뢰도 점수 계산"""
        if not documents:
            return 0.0
        
        # 평균 유사도 점수 기반 신뢰도
        scores = [doc.get('score', 0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        
        # 문서 수에 따른 가중치
        doc_count_weight = min(len(documents) / self.config.max_retrieved_docs, 1.0)
        
        # 최종 신뢰도 (0~1 범위)
        confidence = avg_score * doc_count_weight
        return round(confidence, 2)
    
    async def _evaluate_and_log_quality(self, user_prompt: str, context: str, answer: str):
        """백그라운드에서 RAG 품질 평가 및 MongoDB 로깅"""
        try:
            logger.info("🔍 백그라운드 RAG 품질 평가 시작")
            
            # GPT-4.1을 이용한 품질 평가 및 MongoDB 로깅
            evaluation_result = await evaluate_and_log_rag(
                user_prompt=user_prompt,
                rag_context=context,
                rag_answer=answer
            )
            
            logger.info(
                f"✅ RAG 품질 평가 완료 - "
                f"전체점수: {evaluation_result.overall_score}/10, "
                f"관련성: {evaluation_result.relevance_score}/10, "
                f"정확성: {evaluation_result.accuracy_score}/10, "
                f"완성도: {evaluation_result.completeness_score}/10"
            )
            
        except Exception as e:
            logger.error(f"❌ 백그라운드 RAG 품질 평가 중 오류: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """RAG 시스템 상태 조회"""
        try:
            # 인덱서 정보
            indexer = get_document_indexer()
            indexer_info = indexer.get_indexer_info()
            
            # 벡터 스토어 정보
            vector_info = indexer_info.get('vector_store_info', {})
            
            return {
                "rag_config": {
                    "embedding_model": self.config.embedding_model,
                    "max_retrieved_docs": self.config.max_retrieved_docs,
                    "similarity_threshold": self.config.similarity_threshold,
                    "max_context_length": self.config.max_context_length
                },
                "indexer_status": indexer_info,
                "total_indexed_documents": vector_info.get('total_entities', 0),
                "status": "active" if vector_info.get('total_entities', 0) > 0 else "no_documents"
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 중 오류: {e}")
            return {"status": "error", "message": str(e)}

class SimpleRAGSystem:
    """간단한 RAG 시스템 (LLM 없는 버전)"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
    
    async def ask(self, question: str) -> str:
        """간단한 질의응답"""
        response = await self.rag_system.query(question)
        return response.answer
    
    async def ask_with_details(self, question: str) -> Dict[str, Any]:
        """상세 정보와 함께 질의응답"""
        response = await self.rag_system.query(question)
        return {
            "answer": response.answer,
            "confidence": response.confidence_score,
            "sources": [doc.get('source_file') for doc in response.retrieved_documents],
            "retrieved_count": len(response.retrieved_documents)
        }

# 지연 로딩을 위한 전역 변수
_rag_system = None
_simple_rag = None

def get_rag_system() -> RAGSystem:
    """RAG 시스템 인스턴스 반환 (지연 로딩)"""
    global _rag_system
    
    if _rag_system is None:
        logger.info("기본 RAG 시스템 초기화 중...")
        _rag_system = RAGSystem()
        logger.info("기본 RAG 시스템 초기화 완료")
    
    return _rag_system

def get_simple_rag() -> SimpleRAGSystem:
    """Simple RAG 시스템 인스턴스 반환 (지연 로딩)"""
    global _simple_rag
    
    if _simple_rag is None:
        logger.info("Simple RAG 시스템 초기화 중...")
        _simple_rag = SimpleRAGSystem()
        logger.info("Simple RAG 시스템 초기화 완료")
    
    return _simple_rag

# 편의 함수들
async def rag_query(question: str, max_docs: int = None) -> RAGResponse:
    """RAG 쿼리 편의 함수"""
    system = get_rag_system()
    return await system.query(question, max_docs)

async def simple_ask(question: str) -> str:
    """간단한 질의응답 편의 함수"""
    simple_rag = get_simple_rag()
    return await simple_rag.ask(question)

async def ask_with_context(question: str) -> Dict[str, Any]:
    """컨텍스트 포함 질의응답 편의 함수"""
    simple_rag = get_simple_rag()
    return await simple_rag.ask_with_details(question)

async def get_rag_status() -> Dict[str, Any]:
    """RAG 시스템 상태 조회 편의 함수"""
    try:
        system = get_rag_system()
        return await system.get_system_status()
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "rag_system": "초기화 필요"
        } 