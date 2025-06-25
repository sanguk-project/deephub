"""
RAG 시스템 통합 컴포넌트
검색 증강 생성(Retrieval-Augmented Generation) 기능을 제공
EXAONE-3.5-2.4B-Instruct 기반 품질 평가 및 MongoDB 로깅 포함
"""

import logging
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from service.storage.vector_store import search_similar_documents
from service.storage.rag_logger import evaluate_and_log_rag, quick_rag_score, RAGEvaluationResult
from shared.config.settings import settings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    embedding_model: str = settings.model.embedding_model  # settings에서 임베딩 모델 가져오기
    llm_model: str = settings.model.exaone_model_path  # settings에서 EXAONE 모델 경로 가져오기
    max_retrieved_docs: int = 5 # 최대 몇 개의 문서를 가져올지 지정
    similarity_threshold: float = 0.3 # 검색된 문서 유사도 기준으로 컨텍스트 참조
    max_context_length: int = 4000 # LLM에 전달하는 컨텍스트의 최종 길이
    max_new_tokens: int = 1000  # 답변 시 새로 생성할 최대 토큰 개수
    temperature: float = 0.1  # 생성 온도

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
        
        # 임베딩 모델 초기화 (settings에서 모델명 가져오기)
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.model.embedding_model
        )
        
        # EXAONE 모델과 langchain 파이프라인 초기화
        self._initialize_llm_pipeline()
        
        logger.info(f"RAG 시스템 초기화 완료 - 임베딩: {settings.model.embedding_model}, LLM: {settings.model.exaone_model_path}")
    
    def _initialize_llm_pipeline(self):
        """EXAONE 모델과 langchain 파이프라인 초기화"""
        try:
            local_model_path = settings.model.exaone_model_path
            hf_model_name = settings.model.exaone_model_name
            
            logger.info(f"EXAONE 모델 초기화 시작 - 로컬경로: {local_model_path}")
            
            # 1. 로컬 모델 존재 확인
            if self._check_model_exists(local_model_path):
                logger.info("로컬 모델 발견. 로컬에서 로딩...")
                model_source = local_model_path
            else:
                logger.info("로컬 모델 없음. HuggingFace에서 다운로드 후 저장...")
                
                # 모델 다운로드 및 저장
                if self._download_and_save_model(local_model_path, hf_model_name):
                    logger.info("모델 다운로드 완료. 로컬에서 로딩...")
                    model_source = local_model_path
                else:
                    logger.warning("모델 다운로드 실패. HuggingFace에서 직접 로딩...")
                    model_source = hf_model_name
            
            logger.info(f"모델 소스: {model_source}")
            
            # 2. 토크나이저 로딩
            logger.info("EXAONE 토크나이저 로딩 중...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    trust_remote_code=True,
                    use_fast=True  # Fast tokenizer 사용
                )
                logger.info(f"토크나이저 로딩 완료: {type(tokenizer).__name__}")
            except Exception as e:
                logger.warning(f"Fast tokenizer 로딩 실패, slow tokenizer 시도: {e}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    trust_remote_code=True,
                    use_fast=False  # Slow tokenizer 폴백
                )
                logger.info(f"Slow tokenizer 로딩 완료: {type(tokenizer).__name__}")
            
            # 토크나이저 설정
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"pad_token을 eos_token으로 설정: {tokenizer.eos_token}")
                else:
                    # 폴백: 임의의 토큰 설정
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("pad_token을 [PAD]로 설정")
            
            logger.info(f"토크나이저 설정 완료 - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")
            
            # 3. 모델 로딩
            logger.info("EXAONE 모델 로딩 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # 메모리 효율적 로딩
                load_in_8bit=False,  # 필요시 True로 변경 (양자화)
                load_in_4bit=False   # 필요시 True로 변경 (더 강한 양자화)
            )
            
            # HuggingFace 파이프라인 생성
            logger.info("HuggingFace 파이프라인 생성 중...")
            
            # 토크나이저 ID 검증
            pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            
            if pad_token_id is None:
                logger.warning("pad_token_id가 None입니다. eos_token_id를 사용합니다.")
                pad_token_id = eos_token_id
            
            if eos_token_id is None:
                logger.warning("eos_token_id가 None입니다. 기본값 사용")
                eos_token_id = tokenizer.vocab_size - 1  # 마지막 토큰 ID 사용
            
            logger.info(f"토큰 ID 설정 - pad_token_id: {pad_token_id}, eos_token_id: {eos_token_id}")
            
            # device_map="auto"를 사용한 경우 device 인자 제거
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": True,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "return_full_text": False  # 입력 프롬프트 제외하고 생성된 텍스트만 반환
            }
            
            # GPU가 있고 device_map="auto"를 사용하지 않은 경우에만 device 설정
            if torch.cuda.is_available():
                # accelerate로 로딩된 모델인지 확인
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    logger.info("모델이 accelerate로 로딩됨. device 인자 제거")
                else:
                    pipeline_kwargs["device"] = 0
                    logger.info("GPU에서 실행 - device 설정")
            else:
                pipeline_kwargs["device"] = -1
                logger.info("CPU에서 실행 - device 설정")
            
            hf_pipeline = pipeline("text-generation", **pipeline_kwargs)
            
            logger.info("파이프라인 생성 완료")
            
            # LangChain HuggingFacePipeline 래퍼
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
            
            # 프롬프트 템플릿 정의
            prompt_template = """당신은 회사 문서 기반 AI 어시스턴트입니다. 주어진 문서 내용을 바탕으로 사용자의 질문에 대해 정확하고 정리된 답변을 제공하세요.

            답변 작성 가이드라인:
            1. 핵심 정보를 명확하게 정리하여 답변
            2. 구체적인 내용 (시간, 장소, 연락처, 절차 등)을 빠뜨리지 말고 포함
            3. 불필요한 반복이나 중복 내용은 제거
            4. 사용자가 이해하기 쉽도록 구조화하여 제시
            5. 문서에 없는 내용은 추측하지 말 것

            질문: {question}

            관련 문서 내용:
            {context}

            답변:"""

            self.prompt = PromptTemplate(
                input_variables=["question", "context"],
                template=prompt_template
            )
            
            # 최신 LangChain 방식으로 체인 생성 (prompt | llm | output_parser)
            output_parser = StrOutputParser()
            self.llm_chain = self.prompt | self.llm | output_parser
            
            # 디바이스 정보 확인
            device_info = "auto (accelerate)" if torch.cuda.is_available() and hasattr(model, 'hf_device_map') and model.hf_device_map else pipeline_kwargs.get('device', 'unknown')
            logger.info(f"EXAONE 모델과 LangChain 파이프라인 초기화 완료 (Device: {device_info})")
            
        except Exception as e:
            logger.error(f"EXAONE 모델 초기화 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            
            # 폴백으로 None 설정 (폴백 답변 사용)
            self.llm = None
            self.llm_chain = None
    
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
            
            # 7. 답변 생성 (EXAONE 모델과 LangChain을 활용한 정리된 답변)
            answer = await self._generate_answer(question, context, filtered_docs)
            
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
    
    async def _generate_answer(self, question: str, context: str, documents: List[Dict[str, Any]]) -> str:
        """
        EXAONE 모델과 LangChain을 활용한 답변 생성
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 컨텍스트
            documents (List[Dict[str, Any]]): 검색된 문서들
            
        Returns:
            str: 생성된 답변
        """
        if not documents:
            return "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다. 문서가 인덱싱되어 있는지 확인해주세요."
        
        try:
            # EXAONE 모델이 초기화되어 있는지 확인
            if self.llm_chain is None:
                logger.warning("EXAONE 모델이 초기화되지 않음. 폴백 답변 사용")
                return self._generate_fallback_answer(context, documents)
            
            # LangChain을 통한 답변 생성
            logger.info("EXAONE 모델로 답변 생성 중...")
            
            # 최신 LangChain 방식으로 invoke 사용
            answer = await asyncio.to_thread(
                self.llm_chain.invoke,
                {"question": question, "context": context}
            )
            
            # 답변 후처리
            answer = answer.strip() if isinstance(answer, str) else str(answer).strip()
            
            # 프롬프트 반복이나 불필요한 내용 제거
            if "답변:" in answer:
                answer = answer.split("답변:")[-1].strip()
            
            # 답변이 너무 짧거나 비어있으면 폴백
            if len(answer) < 20:
                logger.warning("생성된 답변이 너무 짧음. 폴백 답변 사용")
                return self._generate_fallback_answer(context, documents)
            
            logger.info("EXAONE 모델 답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.warning(f"EXAONE 모델 답변 생성 중 오류 (폴백 사용): {e}")
            return self._generate_fallback_answer(context, documents)
    
    def _generate_fallback_answer(self, context: str, documents: List[Dict[str, Any]]) -> str:
        """EXAONE 실패 시 사용할 폴백 답변 생성"""
        if not documents:
            return "관련 정보를 찾을 수 없습니다."
        
        # 컨텍스트에서 핵심 정보만 추출
        context_lines = context.split('\n')
        answer_lines = []
        
        for line in context_lines:
            line = line.strip()
            # 문서 헤더 라인은 제외
            if line.startswith('[문서') or not line:
                continue
            # 핵심 내용만 포함
            if len(line) > 10:  # 의미있는 길이의 내용만
                answer_lines.append(line)
        
        if answer_lines:
            # 중복 제거 및 정리
            unique_lines = []
            for line in answer_lines:
                if line not in unique_lines:
                    unique_lines.append(line)
            
            return '\n\n'.join(unique_lines[:5])  # 최대 5개 핵심 정보
        else:
            return "관련 정보를 찾았지만 적절한 답변을 추출할 수 없습니다."
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """검색 품질 점수 계산 (신뢰도 대신 단순 품질 점수)"""
        if not documents:
            return 0.0
        
        # 평균 유사도 점수 기반 품질 점수
        scores = [doc.get('score', 0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        
        # 문서 수에 따른 가중치
        doc_count_weight = min(len(documents) / self.config.max_retrieved_docs, 1.0)
        
        # 최종 품질 점수 (0~1 범위)
        quality_score = avg_score * doc_count_weight
        return round(quality_score, 2)
    
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
            
            # EXAONE 모델 상태 확인
            llm_status = "initialized" if self.llm_chain is not None else "failed"
            
            return {
                "rag_config": {
                    "embedding_model": settings.model.embedding_model,  # settings에서 가져오기
                    "llm_model": settings.model.exaone_model_path,      # settings에서 가져오기
                    "max_retrieved_docs": self.config.max_retrieved_docs,
                    "similarity_threshold": self.config.similarity_threshold,
                    "max_context_length": self.config.max_context_length,
                    "max_new_tokens": self.config.max_new_tokens,
                    "temperature": self.config.temperature
                },
                "llm_status": llm_status,  # LLM 상태 정보 추가
                "indexer_status": indexer_info,
                "total_indexed_documents": vector_info.get('total_entities', 0),
                "status": "active" if vector_info.get('total_entities', 0) > 0 else "no_documents"
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 중 오류: {e}")
            return {"status": "error", "message": str(e)}

    def _download_and_save_model(self, local_path: str, model_name: str):
        """EXAONE 모델을 다운로드하고 로컬에 저장"""
        try:
            import os
            from pathlib import Path
            
            logger.info(f"EXAONE 모델 다운로드 중: {model_name} -> {local_path}")
            
            # 디렉토리 생성
            Path(local_path).mkdir(parents=True, exist_ok=True)
            
            # 토크나이저 다운로드 및 저장
            logger.info("토크나이저 다운로드 중...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            tokenizer.save_pretrained(local_path)
            logger.info(f"토크나이저 저장 완료: {local_path}")
            
            # 모델 다운로드 및 저장
            logger.info("모델 다운로드 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model.save_pretrained(local_path)
            logger.info(f"모델 저장 완료: {local_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 다운로드/저장 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def _check_model_exists(self, local_path: str) -> bool:
        """로컬 모델 존재 여부 확인"""
        try:
            import os
            from pathlib import Path
            
            model_path = Path(local_path)
            
            # 필수 파일들 확인
            required_files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            # pytorch 모델 파일 확인 (여러 가능한 이름)
            model_files = [
                "pytorch_model.bin",
                "model.safetensors", 
                "pytorch_model-00001-of-00001.bin"
            ]
            
            # 필수 파일 존재 확인
            for file in required_files:
                if not (model_path / file).exists():
                    logger.warning(f"필수 파일 없음: {file}")
                    return False
            
            # 모델 파일 중 하나는 존재해야 함
            model_file_exists = any((model_path / file).exists() for file in model_files)
            if not model_file_exists:
                logger.warning("모델 파일 없음")
                return False
            
            logger.info(f"로컬 모델 확인 완료: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 존재 확인 실패: {e}")
            return False

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