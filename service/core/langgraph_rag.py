"""
LangGraph 기반 고급 RAG 시스템
GPT-4.1 검증 + GPT-4.1 최종 답변 생성
스코어 기반 반복 검색 워크플로우
"""
import logging
import json
from typing import List, Dict, Any, TypedDict, Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from sentence_transformers import SentenceTransformer

# MongoDB 로깅 제거됨
from shared.config.settings import settings
from shared.utils import get_openai_client

logger = logging.getLogger(__name__)

@dataclass
class LangGraphRAGConfig:
    """LangGraph RAG 시스템 설정"""
    embedding_model: str = settings.model.embedding_model
    max_retrieved_docs: int = settings.rag.max_retrieved_docs
    similarity_threshold: float = settings.rag.similarity_threshold
    max_context_length: int = settings.rag.max_context_length
    verification_model: str = settings.model.verification_model
    final_answer_model: str = settings.model.final_answer_model
    min_score_threshold: int = settings.rag.min_score_threshold
    max_retry_attempts: int = settings.rag.max_retry_attempts
    openai_api_key: str = settings.api.openai_api_key

class RAGState(TypedDict):
    """RAG 워크플로우 상태"""
    question: str
    retrieved_documents: List[Dict[str, Any]]
    context: str
    rag_score: int  # 0-10 RAG 참조 정확도 점수
    retry_count: int  # 재시도 횟수
    final_answer: str
    confidence_score: float
    sources: List[str]
    reasoning: str
    error: str
    next_action: Literal["retry_search", "generate_final_answer", "end"]  # 다음 액션

class LangGraphRAGSystem:
    """LangGraph 기반 RAG 시스템"""
    
    def __init__(self, config: LangGraphRAGConfig = None):
        self.config = config or LangGraphRAGConfig()
        
        # 임베딩 모델 초기화
        self.embed_model = SentenceTransformer(self.config.embedding_model)
        
        # OpenAI 클라이언트 초기화 (GPT-4.1)
        self.openai_client = get_openai_client(async_mode=False)
        
        # 메모리 저장소 초기화
        self.memory = MemorySaver()
        
        # RAG 워크플로우 그래프 생성
        self.workflow = self._create_workflow()
        
        logger.info("LangGraph RAG 시스템 초기화 완료 (GPT-4.1 + GPT-4.1)")
    
    def _create_workflow(self) -> StateGraph:
        """새로운 RAG 워크플로우 그래프 생성"""
        
        # 상태 그래프 정의
        workflow = StateGraph(RAGState)
        
        # 노드 추가
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("evaluate_rag_quality", self._evaluate_rag_quality)
        workflow.add_node("improve_search", self._improve_search)
        workflow.add_node("generate_final_answer", self._generate_final_answer)
        
        # 엣지 정의 (워크플로우 흐름)
        workflow.add_edge(START, "retrieve_documents")
        workflow.add_edge("retrieve_documents", "evaluate_rag_quality")
        
        # 조건부 엣지: 스코어에 따른 분기
        workflow.add_conditional_edges(
            "evaluate_rag_quality",
            self._decide_next_step,
            {
                "retry_search": "improve_search",
                "generate_final_answer": "generate_final_answer",
                "end": END
            }
        )
        
        # 개선된 검색 후 다시 평가
        workflow.add_edge("improve_search", "evaluate_rag_quality")
        workflow.add_edge("generate_final_answer", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """문서 검색 단계"""
        try:
            question = state["question"]
            retry_count = state.get("retry_count", 0)
            
            # 질문 임베딩 생성
            question_embedding = self.embed_model.encode(question).tolist()
            
            # 재시도 시 검색 전략 조정
            search_limit = self.config.max_retrieved_docs * (3 + retry_count)  # 재시도 시 더 많은 문서 검색
            
            # 유사 문서 검색
            from service.storage.vector_store import faiss_store
            import asyncio
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                similar_docs = loop.run_until_complete(
                    faiss_store.search_similar(
                        query_embedding=question_embedding,
                        limit=search_limit
                    )
                )
            finally:
                loop.close()
            
            # 중복 제거 및 필터링
            filtered_docs = self._filter_and_deduplicate(similar_docs)
            
            # 컨텍스트 생성
            context = self._build_context(filtered_docs)
            
            state["retrieved_documents"] = filtered_docs
            state["context"] = context
            state["sources"] = [doc.get("source_file", "") for doc in filtered_docs]
            
            logger.info(f"문서 검색 완료: {len(filtered_docs)}개 문서 (재시도 {retry_count}회)")
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {e}")
            state["error"] = f"문서 검색 실패: {str(e)}"
        
        return state
    
    def _evaluate_rag_quality(self, state: RAGState) -> RAGState:
        """GPT-4.1로 RAG 품질 평가 (0-10점)"""
        try:
            question = state["question"]
            context = state["context"]
            
            if not context:
                state["rag_score"] = 0
                state["reasoning"] = "검색된 컨텍스트가 없습니다."
                state["next_action"] = "retry_search"
                return state
            
            # GPT-4.1로 RAG 참조 정확도 평가
            evaluation_prompt = f"""
다음 질문에 대해 제공된 컨텍스트가 얼마나 정확하고 관련성이 높은지 0-10점으로 평가해주세요.

질문: {question}

제공된 컨텍스트:
{context}

평가 기준:
1. 질문과 컨텍스트의 관련성 (0-3점)
2. 컨텍스트의 완성도 및 충분성 (0-3점)  
3. 정보의 정확성 및 신뢰성 (0-2점)
4. 답변 생성 가능성 (0-2점)

다음 JSON 형식으로 응답해주세요:
{{
    "score": 0-10,
    "reasoning": "상세한 평가 이유",
    "missing_aspects": "부족한 부분이 있다면 설명",
    "improvement_suggestions": "개선 제안사항"
}}
"""

            response = self.openai_client.chat.completions.create(
                model=self.config.verification_model,
                messages=[
                    {"role": "system", "content": "당신은 RAG 시스템의 품질을 평가하는 전문가입니다. 주어진 질문과 컨텍스트를 분석하여 정확한 점수와 이유를 제공해주세요."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # JSON 응답 파싱
            try:
                evaluation_result = json.loads(response.choices[0].message.content)
                rag_score = int(evaluation_result.get("score", 0))
                reasoning = evaluation_result.get("reasoning", "")
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 실패 시 기본값
                rag_score = 3
                reasoning = f"평가 결과 파싱 실패: {response.choices[0].message.content[:200]}"
            
            state["rag_score"] = rag_score
            state["reasoning"] = reasoning
            
            # 다음 단계 결정
            if rag_score >= self.config.min_score_threshold:
                state["next_action"] = "generate_final_answer"
            elif state.get("retry_count", 0) >= self.config.max_retry_attempts:
                state["next_action"] = "end"
                state["error"] = f"최대 재시도 횟수 초과 (최종 점수: {rag_score})"
            else:
                state["next_action"] = "retry_search"
            
            logger.info(f"RAG 품질 평가 완료: {rag_score}/10점")
            
        except Exception as e:
            logger.error(f"RAG 품질 평가 중 오류: {e}")
            state["rag_score"] = 0
            state["reasoning"] = f"평가 실패: {str(e)}"
            state["next_action"] = "retry_search"
        
        return state
    
    def _decide_next_step(self, state: RAGState) -> str:
        """다음 단계 결정"""
        return state.get("next_action", "end")
    
    def _improve_search(self, state: RAGState) -> RAGState:
        """검색 개선 및 재시도"""
        try:
            retry_count = state.get("retry_count", 0) + 1
            state["retry_count"] = retry_count
            
            logger.info(f"검색 개선 시도 {retry_count}회")
            
            # 기존 검색 결과를 기반으로 질문 확장/개선
            question = state["question"]
            previous_reasoning = state.get("reasoning", "")
            
            # GPT-4.1로 질문 개선 제안
            improvement_prompt = f"""
다음 질문에 대한 검색 결과가 부족했습니다. 더 나은 검색을 위해 질문을 개선하거나 확장해주세요.

원본 질문: {question}
부족한 이유: {previous_reasoning}

개선된 질문이나 추가 검색 키워드를 제안해주세요:
"""
            
            try:
                improvement_response = self.openai_client.chat.completions.create(
                    model=self.config.verification_model,
                    messages=[
                        {"role": "system", "content": "당신은 검색 쿼리 개선 전문가입니다. 검색 결과가 부족한 질문을 분석하여 더 나은 검색을 위한 제안을 해주세요."},
                        {"role": "user", "content": improvement_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                # 개선된 질문을 임시로 저장 (실제 구현에서는 더 정교한 방법 사용)
                improved_question = improvement_response.choices[0].message.content.strip()
                logger.info(f"개선된 검색 전략: {improved_question[:100]}...")
                
            except Exception as e:
                logger.warning(f"질문 개선 실패: {e}")
            
        except Exception as e:
            logger.error(f"검색 개선 중 오류: {e}")
            state["error"] = f"검색 개선 실패: {str(e)}"
        
        return state
    
    def _generate_final_answer(self, state: RAGState) -> RAGState:
        """GPT-4.1로 최종 답변 생성"""
        try:
            question = state["question"]
            context = state["context"]
            rag_score = state.get("rag_score", 0)
            
            # GPT-4.1로 최종 답변 생성
            final_prompt = f"""
다음 컨텍스트를 바탕으로 사용자의 질문에 정확하고 상세하게 답변해주세요.

질문: {question}

컨텍스트 (RAG 품질 점수: {rag_score}/10):
{context}

답변 가이드라인:
1. 컨텍스트에 있는 정보만을 사용하여 답변하세요
2. 정확한 정보를 제공하고, 불확실한 부분은 명시하세요  
3. 한국어로 자연스럽고 이해하기 쉽게 답변하세요
4. 필요시 구체적인 예시나 설명을 포함하세요
5. 컨텍스트가 부족한 경우 그 점을 명시하고 제한적인 답변임을 알려주세요

답변:"""

            response = self.openai_client.chat.completions.create(
                model=self.config.final_answer_model,
                messages=[
                    {"role": "system", "content": "당신은 정확하고 도움이 되는 AI 어시스턴트입니다. 주어진 컨텍스트만을 사용하여 답변하세요."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            final_answer = response.choices[0].message.content
            state["final_answer"] = final_answer
            
            # 신뢰도 점수 계산 (RAG 점수 기반)
            confidence_score = min(rag_score / 10.0, 1.0)
            state["confidence_score"] = confidence_score
            
            logger.info(f"GPT-4.1 최종 답변 생성 완료 (RAG 점수: {rag_score}/10)")
            
        except Exception as e:
            logger.error(f"최종 답변 생성 중 오류: {e}")
            state["error"] = f"최종 답변 생성 실패: {str(e)}"
            state["final_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        
        return state
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """검색된 문서들로부터 컨텍스트 구성"""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = doc.get('text', '')
            source_file = doc.get('source_file', 'unknown')
            
            if current_length + len(doc_text) > self.config.max_context_length:
                break
            
            context_part = f"[문서 {i+1} - {source_file}]\n{doc_text}\n"
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def query(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """새로운 RAG 쿼리 실행"""
        import uuid
        
        # 세션 ID 생성 (제공되지 않은 경우)
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            # 초기 상태 생성
            initial_state = RAGState(
                question=question,
                retrieved_documents=[],
                context="",
                rag_score=0,
                retry_count=0,
                final_answer="",
                confidence_score=0.0,
                sources=[],
                reasoning="",
                error=None,
                next_action="retry_search"
            )
            
            # 워크플로우 실행
            config = {"configurable": {"thread_id": session_id}}
            final_state = self.workflow.invoke(initial_state, config)
            
            # 결과 정리
            result = {
                "session_id": session_id,
                "answer": final_state.get("final_answer", "답변을 생성하지 못했습니다."),
                "confidence": final_state.get("confidence_score", 0.0),
                "sources": final_state.get("sources", []),
                "rag_score": final_state.get("rag_score", 0),
                "reasoning": final_state.get("reasoning", ""),
                "retrieved_count": len(final_state.get("retrieved_documents", [])),
                "retry_count": final_state.get("retry_count", 0),
                "error": final_state.get("error")
            }
            
            # 로깅 기능 제거됨
            
            return result
            
        except Exception as e:
            logger.error(f"RAG 쿼리 실행 중 오류: {e}")
            
            # 로깅 기능 제거됨
            
            return {
                "session_id": session_id,
                "answer": "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "rag_score": 0,
                "reasoning": f"시스템 오류: {str(e)}",
                "retrieved_count": 0,
                "retry_count": 0,
                "error": str(e)
            }
    
    def _filter_and_deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 필터링 및 중복 제거"""
        if not documents:
            return []
        
        # 유사도 임계값 필터링
        filtered_docs = [
            doc for doc in documents 
            if doc.get('score', 0) >= self.config.similarity_threshold
        ]
        
        # source_file별 중복 제거
        file_groups = {}
        for doc in filtered_docs:
            source_file = doc.get('source_file', 'unknown')
            if source_file not in file_groups:
                file_groups[source_file] = []
            file_groups[source_file].append(doc)
        
        deduplicated = []
        for source_file, docs in file_groups.items():
            # 점수순으로 정렬하고 최고 점수 문서 선택
            docs.sort(key=lambda x: x.get('score', 0), reverse=True)
            representative_doc = docs[0].copy()
            
            # 상위 3개 청크의 텍스트 결합
            combined_texts = []
            for doc in docs[:3]:
                text = doc.get('text', '').strip()
                if text and text not in combined_texts:
                    combined_texts.append(text)
            
            representative_doc['text'] = '\n\n'.join(combined_texts)
            representative_doc['combined_chunks'] = len(docs)
            deduplicated.append(representative_doc)
        
        # 점수순 정렬 후 제한
        deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
        return deduplicated[:self.config.max_retrieved_docs]
    
    def _calculate_initial_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """초기 신뢰도 점수 계산"""
        if not documents:
            return 0.0
        
        scores = [doc.get('score', 0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        doc_count_factor = min(len(documents) / self.config.max_retrieved_docs, 1.0)
        
        return (avg_score * 0.4 + max_score * 0.4 + doc_count_factor * 0.2)

# 지연 로딩을 위한 전역 변수
_langgraph_rag_system = None

def get_langgraph_system() -> LangGraphRAGSystem:
    """LangGraph RAG 시스템 인스턴스 반환 (지연 로딩)"""
    global _langgraph_rag_system
    
    if _langgraph_rag_system is None:
        logger.info("LangGraph RAG 시스템 초기화 중...")
        _langgraph_rag_system = LangGraphRAGSystem()
        logger.info("LangGraph RAG 시스템 초기화 완료")
    
    return _langgraph_rag_system

# 편의 함수
async def langgraph_ask_with_context(question: str) -> Dict[str, Any]:
    """LangGraph RAG 시스템을 통한 질의응답"""
    system = get_langgraph_system()
    return system.query(question)

async def get_langgraph_rag_status() -> Dict[str, Any]:
    """LangGraph RAG 시스템 상태 조회"""
    try:
        system = get_langgraph_system()
        return {
            "status": "active",
            "model": "LangGraph + GPT-4.1 + GPT-4.1 하이브리드 (검증과 답변 생성 모두 GPT-4.1)",
            "embedding_model": system.config.embedding_model,
            "verification_model": system.config.verification_model,
            "final_answer_model": system.config.final_answer_model,
            "max_retrieved_docs": system.config.max_retrieved_docs,
            "similarity_threshold": system.config.similarity_threshold,
            "min_score_threshold": system.config.min_score_threshold,
            "max_retry_attempts": system.config.max_retry_attempts
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": "LangGraph + GPT-4.1 (초기화 필요)"
        } 