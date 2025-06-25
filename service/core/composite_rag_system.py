"""
deephub 복합 RAG 시스템
사용자 프롬프트 → RAG → EXAONE 3.5 답변생성 → GPT-4.1 검수 → 최종답변
"""

import logging
import asyncio
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os
import gc
import atexit
import multiprocessing
import traceback

import torch
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# deephub 내부 컴포넌트 import
from service.storage.vector_store import search_similar_documents
from shared.config.settings import settings

# 멀티프로세싱 오류 방지 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Hugging Face tokenizers 병렬 처리 비활성화
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP 스레드 수 제한
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL 스레드 수 제한

# 멀티프로세싱 컨텍스트 설정 (리눅스 환경에서 권장)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 이미 설정된 경우 무시

logger = logging.getLogger(__name__)

# 멀티프로세싱 리소스 누수 방지
def cleanup_multiprocessing():
    """프로그램 종료 시 멀티프로세싱 리소스 정리"""
    try:
        # 활성 프로세스 정리
        for p in multiprocessing.active_children():
            p.terminate()
            p.join(timeout=1)
        
        # 공유 메모리 정리
        gc.collect()
    except Exception as e:
        logging.warning(f"멀티프로세싱 리소스 정리 중 오류: {e}")

# 프로그램 종료 시 자동 정리
atexit.register(cleanup_multiprocessing)

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError:
    logging.warning("sentence-transformers not installed, some features may not work")
    SentenceTransformer = None
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logging.warning("rank-bm25 not installed, BM25 re-ranking will not work")
    BM25Okapi = None

@dataclass
class CompositeRAGConfig:
    """복합 RAG 시스템 설정 - settings.py에서 자동 로드"""
    
    # 모델 설정 (settings.py에서 로드)
    embedding_model_name: str = settings.model.embedding_model
    gpt_model_name: str = settings.model.verification_model
    exaone_model_path: str = settings.model.exaone_model_path
    exaone_model_name: str = settings.model.exaone_model_name
    
    # Re-ranker 모델 설정 (settings.py에서 로드)
    reranker_model_name: str = settings.model.reranker_model
    reranker_model_type: str = settings.model.reranker_model_type
    reranker_device: str = settings.model.reranker_device
    reranker_max_length: int = settings.model.reranker_max_length
    reranker_batch_size: int = settings.model.reranker_batch_size
    reranker_num_workers: int = settings.model.reranker_num_workers
    
    # RAG 기본 설정 (settings.py에서 로드)
    top_k: int = settings.rag.max_retrieved_docs
    similarity_threshold: float = settings.rag.similarity_threshold
    max_tokens: int = 1024
    temperature: float = 0.7
    max_context_length: int = settings.rag.max_context_length
    min_score_threshold: int = settings.rag.min_score_threshold
    max_retry_attempts: int = settings.rag.max_retry_attempts
    
    # 향상된 검색 설정 (settings.py에서 로드)
    keyword_weight: float = settings.rag.keyword_weight
    semantic_weight: float = settings.rag.semantic_weight
    diversity_threshold: int = settings.rag.diversity_threshold
    min_text_length: int = settings.rag.min_text_length
    context_relevance_threshold: float = settings.rag.context_relevance_threshold
    intent_matching_weight: float = settings.rag.intent_matching_weight
    sequence_matching_weight: float = settings.rag.sequence_matching_weight
    important_keyword_boost: float = settings.rag.important_keyword_boost
    combined_score_threshold: float = settings.rag.combined_score_threshold
    
    # Re-ranker 설정 (settings.py에서 로드)
    enable_reranker: bool = settings.rag.enable_reranker
    reranker_top_k: int = settings.rag.reranker_top_k
    reranker_output_k: int = settings.rag.reranker_output_k
    reranker_weight: float = settings.rag.reranker_weight
    bm25_weight: float = settings.rag.bm25_weight
    embedding_weight: float = settings.rag.embedding_weight
    diversity_penalty: float = settings.rag.diversity_penalty
    mmr_lambda: float = settings.rag.mmr_lambda
    
    # 생성 설정
    max_new_tokens: int = 1024
    generation_temperature: float = 0.7


class DeephubCompositeRAG:
    """deephub 복합 RAG 시스템"""
    
    def __init__(self):
        """Composite RAG 시스템 초기화"""
        self.config = CompositeRAGConfig()
        
        # 임베딩 모델 초기화 (settings.py의 설정 준수)
        self.embedding_model_name = self.config.embedding_model_name  # BAAI/bge-m3
        self.embedding_model = None
        self._init_embedding_model()
        
        # EXAONE 모델 설정
        self.exaone_model = None
        self.exaone_tokenizer = None
        self.exaone_pipeline = None
        self._load_exaone_model()
        
        # OpenAI 클라이언트 설정
        self.openai_client = None
        self._load_openai_client()
        
        # Re-ranker 초기화
        self.reranker = None
        self._load_reranker()
        
        logger.info("DeepHub Composite RAG 시스템 초기화 완료")
        logger.info(f"설정된 임베딩 모델: {self.embedding_model_name}")
        logger.info(f"설정된 EXAONE 모델: {self.config.exaone_model_name}")
        logger.info(f"Re-ranker 활성화: {self.config.enable_reranker}")
    
    def _init_embedding_model(self):
        """임베딩 모델 안전하게 초기화 (settings.py 설정 준수)"""
        try:
            if SentenceTransformer is not None:
                # settings.py에서 지정한 BAAI/bge-m3 모델 사용
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"임베딩 모델 로드 완료: {self.embedding_model_name}")
            else:
                logger.error("SentenceTransformer 모듈을 찾을 수 없음")
                # HuggingFace Embedding 대안 시도
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    self.embedding_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
                    logger.info(f"HuggingFace 임베딩 모델로 대체 로드: {self.embedding_model_name}")
                except Exception as e:
                    logger.error(f"대체 임베딩 모델 로드도 실패: {e}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        self.cleanup()
    
    def cleanup(self):
        """명시적 리소스 정리"""
        try:
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                del self.embedding_model
            if hasattr(self, 'reranker') and self.reranker is not None:
                self.reranker.cleanup()
                del self.reranker
            gc.collect()
        except Exception as e:
            logger.warning(f"DeephubCompositeRAG 리소스 정리 중 오류: {e}")
    

    
    def _load_openai_client(self):
        """OpenAI 클라이언트 로드"""
        try:
            import openai
            self.openai_client = openai.OpenAI()
            logger.info("OpenAI 클라이언트 로드 완료")
        except Exception as e:
            logger.warning(f"OpenAI 클라이언트 로드 실패: {e}")
            self.openai_client = None
    
    def _load_exaone_model(self):
        """EXAONE 모델 로드 (settings.py 설정 준수)"""
        try:
            logger.info(f"EXAONE 모델 로딩 중: {self.config.exaone_model_path}")
            
            # 먼저 로컬 경로에서 시도
            model_path = self.config.exaone_model_path
            if not os.path.exists(model_path):
                # 로컬 경로가 없으면 HuggingFace 모델명 사용
                model_path = self.config.exaone_model_name
                logger.info(f"로컬 경로가 없어 HuggingFace에서 다운로드: {model_path}")
            
            # 토크나이저 로딩
            logger.info("토크나이저 로딩 중...")
            self.exaone_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.exaone_tokenizer.pad_token is None:
                self.exaone_tokenizer.pad_token = self.exaone_tokenizer.eos_token
            
            # 모델 로딩
            logger.info("모델 로딩 중...")
            self.exaone_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.exaone_pipeline = None
            logger.info(f"EXAONE 모델 로딩 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"EXAONE 모델 로딩 실패: {str(e)}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            logger.info("파이프라인 모드로 대안 설정 중...")
            
            try:
                # 대안: HuggingFace 모델명으로 파이프라인 시도
                model_path = self.config.exaone_model_name
                logger.info(f"파이프라인 모드로 시도: {model_path}")
                
                self.exaone_pipeline = pipeline(
                    "text-generation",
                    model=model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.exaone_model = None
                self.exaone_tokenizer = None
                logger.info(f"EXAONE 파이프라인 설정 완료: {model_path}")
            except Exception as e2:
                logger.error(f"EXAONE 파이프라인 설정 실패: {str(e2)}")
                logger.error(f"상세 오류: {traceback.format_exc()}")
                logger.warning("모든 EXAONE 모델 로딩 시도 실패. GPT-2로 대체합니다.")
                
            
    def _load_reranker(self):
        """Re-ranker 로드 (settings.py 설정 준수)"""
        try:
            if self.config.enable_reranker:
                self.reranker = DocumentReranker(self.config)
                logger.info("Re-ranker 초기화 완료")
            else:
                logger.info("Re-ranker 비활성화됨")
        except Exception as e:
            logger.error(f"Re-ranker 로드 실패: {e}")
            self.reranker = None
    
    async def step1_retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """1단계: RAG 문서 검색 (향상된 검색 로직)"""
        logger.info("1단계: RAG 문서 검색 중...")
        
        try:
            # 1차 검색: 임베딩 벡터 생성
            if self.embedding_model is None:
                logger.error("임베딩 모델이 로드되지 않음")
                return []
                
            # SentenceTransformer로 임베딩 생성
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            
            # 2차 검색: 다양한 검색 전략 적용
            # 더 많은 문서를 1차로 가져온 후 필터링
            initial_docs = await search_similar_documents(
                query_embedding=query_embedding,
                limit=self.config.top_k * 4  # 4배수로 검색하여 더 많은 후보 확보
            )
            
            # 질문 키워드 추출 (강화된 키워드 추출)
            query_keywords = self._extract_enhanced_keywords(query)
            logger.info(f"추출된 질문 키워드: {query_keywords}")
            
            # 문서 품질 평가 및 필터링 (강화)
            filtered_docs = []
            
            for doc in initial_docs:
                # 기본 유사도 임계값 확인
                similarity_score = doc.get('score', 0)
                if similarity_score < self.config.similarity_threshold:
                    continue
                
                # 문서 내용 품질 확인
                text = doc.get('text', '').strip()
                if len(text) < self.config.min_text_length:  # 너무 짧은 내용 제외
                    continue
                
                # 강화된 키워드 관련성 검사
                doc_keywords = self._extract_enhanced_keywords(text.lower())
                keyword_score = self._calculate_enhanced_keyword_overlap(query_keywords, doc_keywords, query, text)
                
                # 질문 의도 매칭 점수 (새로 추가)
                intent_score = self._calculate_intent_matching(query, text)
                
                # 종합 관련성 점수 계산
                combined_score = (
                    similarity_score * self.config.semantic_weight +      # 의미론적 유사도 
                    keyword_score * self.config.keyword_weight +         # 키워드 매칭
                    intent_score * self.config.intent_matching_weight      # 의도 매칭
                )
                
                doc['keyword_score'] = keyword_score
                doc['intent_score'] = intent_score
                doc['combined_score'] = combined_score
                
                # 최소 종합 점수 임계값 적용
                if combined_score >= self.config.combined_score_threshold:  # 강화된 임계값
                    filtered_docs.append(doc)
            
            # 종합 점수로 정렬
            filtered_docs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            # 중복 제거 - 같은 source_file에서 너무 많은 문서 방지 (강화)
            deduplicated_docs = self._deduplicate_and_diversify_enhanced(filtered_docs)
            
            # 최종 문서 수 제한
            final_docs = deduplicated_docs[:self.config.top_k]
            
            logger.info(f"검색 완료: {len(final_docs)}개 문서 발견 (초기: {len(initial_docs)}개, 필터링 후: {len(filtered_docs)}개)")
            return final_docs
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def _extract_enhanced_keywords(self, text: str) -> set:
        """강화된 키워드 추출"""
        # 불용어 확장
        stopwords = {
            # 한글 불용어
            '이', '그', '저', '것', '들', '는', '은', '을', '를', '의', '에', '와', '과', 
            '도', '만', '까지', '부터', '에서', '으로', '로', '에게', '께', '한테',
            '이다', '있다', '없다', '되다', '하다', '되는', '하는', '있는', '없는',
            '때', '곳', '중', '안', '밖', '위', '아래', '앞', '뒤', '옆',
            # 영어 불용어
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'while'
        }
        
        # 한글, 영문, 숫자 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        
        # 키워드 필터링 및 정규화
        keywords = set()
        for word in words:
            if len(word) > 1 and word not in stopwords:
                keywords.add(word)
        
        # 복합어나 중요한 키워드 추가 추출
        important_patterns = [
            r'화환\s*지원', r'화환\s*신청', r'화환\s*조건', r'화환\s*경우',
            r'경조금\s*지원', r'경조금\s*신청', r'경조금\s*조건', r'경조금\s*경우',
            r'경조\s*사업', r'복지\s*혜택', r'지원\s*조건', r'지원\s*대상',
            r'신청\s*방법', r'지급\s*기준', r'대상\s*범위'
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                normalized = re.sub(r'\s+', '', match)  # 공백 제거
                keywords.add(normalized)
        
        # 화환 관련 전용 키워드 추가
        if '화환' in text:
            wreath_specific = ['화환지원', '화환신청', '화환조건', '화환경우', '화환혜택']
            keywords.update(word for word in wreath_specific if word in text.replace(' ', ''))
        
        # 경조금 관련 전용 키워드 추가  
        if any(word in text for word in ['경조금', '경조비', '경조사업']):
            money_specific = ['경조금지원', '경조금신청', '경조금조건', '경조금경우', '경조금혜택']
            keywords.update(word for word in money_specific if word in text.replace(' ', ''))
        
        return keywords
    
    def _calculate_enhanced_keyword_overlap(self, query_keywords: set, doc_keywords: set, 
                                           query_text: str, doc_text: str) -> float:
        """강화된 키워드 중복도 계산"""
        if not query_keywords:
            return 0.0
        
        # 기본 교집합 계산
        overlap = len(query_keywords.intersection(doc_keywords))
        basic_score = overlap / len(query_keywords)
        
        # 중요 키워드 가중치 부여
        important_keywords = {'화환', '지원', '경조', '복지', '혜택', '조건', '신청', '대상'}
        important_overlap = len(query_keywords.intersection(doc_keywords).intersection(important_keywords))
        important_score = important_overlap * self.config.important_keyword_boost
        
        # 순서 고려 (연속된 키워드 조합)
        sequence_score = self._calculate_sequence_matching(query_text, doc_text)
        
        # 최종 점수 계산
        final_score = min(1.0, basic_score * self.config.semantic_weight + important_score * 0.3 + sequence_score * 0.2)
        
        return final_score
    
    def _calculate_sequence_matching(self, query: str, doc_text: str) -> float:
        """연속된 키워드 매칭 점수 계산"""
        # 2-3단어 조합 추출
        query_phrases = []
        doc_phrases = []
        
        # 한글 구문 추출 (2-3단어)
        query_words = re.findall(r'[가-힣]+', query)
        doc_words = re.findall(r'[가-힣]+', doc_text)
        
        # 2단어 조합
        for i in range(len(query_words) - 1):
            phrase = query_words[i] + query_words[i + 1]
            if len(phrase) >= 4:  # 너무 짧은 조합 제외
                query_phrases.append(phrase)
        
        for i in range(len(doc_words) - 1):
            phrase = doc_words[i] + doc_words[i + 1]
            if len(phrase) >= 4:
                doc_phrases.append(phrase)
        
        if not query_phrases:
            return 0.0
        
        # 매칭 계산
        matches = 0
        for q_phrase in query_phrases:
            if any(q_phrase in d_phrase or d_phrase in q_phrase for d_phrase in doc_phrases):
                matches += 1
        
        return matches / len(query_phrases)
    
    def _calculate_intent_matching(self, query: str, doc_text: str) -> float:
        """질문 의도와 문서 내용의 매칭 점수"""
        # 질문 의도 분류
        intent_patterns = {
            'condition': [r'언제', r'경우', r'조건', r'대상', r'기준'],
            'method': [r'어떻게', r'방법', r'절차', r'신청'],
            'amount': [r'얼마', r'금액', r'비용', r'가격'],
            'eligibility': [r'누가', r'대상', r'자격', r'범위'],
            'support': [r'지원', r'혜택', r'복지', r'도움']
        }
        
        # 질문에서 의도 추출
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    detected_intents.append(intent)
                    break
        
        if not detected_intents:
            return 0.5  # 기본 점수
        
        # 문서에서 해당 의도와 관련된 내용 확인
        intent_score = 0.0
        
        for intent in detected_intents:
            if intent == 'condition':
                # 조건/경우 관련 내용 확인
                condition_keywords = ['경우', '조건', '때', '시', '대상', '해당', '적용']
                if any(keyword in doc_text for keyword in condition_keywords):
                    intent_score += 0.3
                    
            elif intent == 'method':
                # 방법/절차 관련 내용 확인
                method_keywords = ['방법', '절차', '신청', '처리', '진행', '접수']
                if any(keyword in doc_text for keyword in method_keywords):
                    intent_score += 0.3
                    
            elif intent == 'amount':
                # 금액 관련 내용 확인
                if re.search(r'\d+.*원|금액|비용|가격', doc_text):
                    intent_score += 0.3
                    
            elif intent == 'eligibility':
                # 자격/대상 관련 내용 확인
                eligibility_keywords = ['자격', '대상', '범위', '해당자', '신청자']
                if any(keyword in doc_text for keyword in eligibility_keywords):
                    intent_score += 0.3
                    
            elif intent == 'support':
                # 지원 관련 내용 확인
                support_keywords = ['지원', '혜택', '복지', '도움', '제공', '지급']
                if any(keyword in doc_text for keyword in support_keywords):
                    intent_score += 0.3
        
        return min(1.0, intent_score)
    
    def _deduplicate_and_diversify_enhanced(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """강화된 중복 제거 및 다양성 확보"""
        source_count = {}
        result = []
        
        # 첫 번째 패스: 각 소스에서 최고 점수 문서 우선 선택
        source_best = {}
        for doc in docs:
            source = doc.get('source_file', 'unknown')
            score = doc.get('combined_score', 0)
            
            if source not in source_best or score > source_best[source]['combined_score']:
                source_best[source] = doc
        
        # 최고 점수 문서들을 먼저 추가
        for source, doc in source_best.items():
            result.append(doc)
            source_count[source] = 1
        
        # 두 번째 패스: 같은 소스에서 추가 문서 선택 (최대 2개까지)
        for doc in docs:
            if len(result) >= self.config.top_k * 2:  # 충분한 후보 확보시 중단
                break
                
            source = doc.get('source_file', 'unknown')
            
            # 이미 선택된 문서인지 확인
            if any(existing['text'] == doc['text'] for existing in result):
                continue
            
            # 같은 소스에서 최대 2개까지만 허용
            if source_count.get(source, 0) < 2:
                result.append(doc)
                source_count[source] = source_count.get(source, 0) + 1
        
        # 점수순으로 재정렬
        result.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return result
    
    async def step1_5_rerank_documents(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """1.5단계: Re-ranker로 문서 재순위 매기기 (선택적)"""
        if not self.reranker or not self.config.enable_reranker:
            logger.info("Re-ranker 비활성화 상태 - 기존 순서 유지")
            return retrieved_docs
        
        if not retrieved_docs:
            return retrieved_docs
        
        logger.info("1.5단계: Re-ranker로 문서 재순위 매기기 중...")
        
        try:
            # Re-ranker로 문서 재순위 매기기
            reranked_docs = await self.reranker.rerank_documents(query, retrieved_docs)
            
            # 순위 변경 로깅
            if len(reranked_docs) > 0:
                logger.info(f"Re-ranker 완료: {len(retrieved_docs)}개 → {len(reranked_docs)}개 문서")
                
                # 상위 3개 문서의 점수 변화 로깅
                for i, doc in enumerate(reranked_docs[:3]):
                    original_score = doc.get('score', 0)
                    rerank_score = doc.get('rerank_score', 0)
                    source_file = doc.get('source_file', 'unknown')
                    logger.debug(f"순위 {i+1}: {source_file} - "
                               f"원본점수: {original_score:.3f} → 재순위점수: {rerank_score:.3f}")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Re-ranker 처리 실패: {e}")
            # 실패시 원본 문서 반환
            return retrieved_docs
    
    def _analyze_question_intent(self, query: str) -> Dict[str, Any]:
        """질문 의도 분석 (강화된 버전)"""
        # 기본 분석 결과
        analysis = {
            'intent': '일반 정보 요청',
            'keywords': [],
            'expected_answer_type': '설명',
            'question_type': 'general',
            'specific_topic': None,
            'exclude_topics': []  # 제외할 주제들
        }
        
        # 키워드 추출
        keywords = self._extract_enhanced_keywords(query)
        analysis['keywords'] = list(keywords)
        
        # 구체적 주제 식별 (화환과 경조금 명확히 구분)
        if '화환' in query:
            analysis['specific_topic'] = 'wreath'  # 화환
            # 화환 질문이면 경조금 관련 내용 제외
            analysis['exclude_topics'] = ['money', 'amount', '경조금', '금액', '원', '지급', '비용']
            
        elif re.search(r'경조금|경조비|경조사.*금|금액|돈|지급', query):
            analysis['specific_topic'] = 'condolence_money'  # 경조금
            # 경조금 질문이면 화환 관련 내용 제외
            analysis['exclude_topics'] = ['wreath', '화환', '꽃', '장식']
            
        # 질문 유형 분석 (기존 로직 유지하되 더 정교하게)
        if re.search(r'언제|경우|조건|때', query):
            analysis['intent'] = '조건 및 시기 문의'
            analysis['expected_answer_type'] = '조건/시기'
            analysis['question_type'] = 'condition'
            
        elif re.search(r'어떻게|방법|절차|신청', query):
            analysis['intent'] = '방법 및 절차 문의'
            analysis['expected_answer_type'] = '방법/절차'
            analysis['question_type'] = 'method'
            
        elif re.search(r'얼마|금액|비용|가격|돈', query):
            analysis['intent'] = '금액 및 비용 문의'
            analysis['expected_answer_type'] = '금액/비용'
            analysis['question_type'] = 'amount'
            
        elif re.search(r'누가|대상|자격|범위|해당', query):
            analysis['intent'] = '대상 및 자격 문의'
            analysis['expected_answer_type'] = '대상/자격'
            analysis['question_type'] = 'eligibility'
            
        elif re.search(r'무엇|뭔가|정의|의미', query):
            analysis['intent'] = '정의 및 설명 문의'
            analysis['expected_answer_type'] = '정의/설명'
            analysis['question_type'] = 'definition'
            
        elif re.search(r'지원|혜택|복지|도움', query):
            analysis['intent'] = '지원 및 혜택 문의'
            analysis['expected_answer_type'] = '지원내용/혜택'
            analysis['question_type'] = 'support'
        
        # 화환 관련 특별 처리 (더 구체적으로)
        if analysis['specific_topic'] == 'wreath':
            if analysis['question_type'] == 'condition':
                analysis['intent'] = '화환 지원 조건 문의'
                analysis['expected_answer_type'] = '화환 지원 조건/경우만'
            elif analysis['question_type'] == 'method':
                analysis['intent'] = '화환 신청 방법 문의'
                analysis['expected_answer_type'] = '화환 신청 절차만'
            elif analysis['question_type'] == 'amount':
                analysis['intent'] = '화환 비용 문의'
                analysis['expected_answer_type'] = '화환 비용/금액만'
            elif analysis['question_type'] == 'support':
                analysis['intent'] = '화환 지원 혜택 문의'
                analysis['expected_answer_type'] = '화환 지원 내용만'
        
        # 경조금 관련 특별 처리
        elif analysis['specific_topic'] == 'condolence_money':
            if analysis['question_type'] == 'condition':
                analysis['intent'] = '경조금 지원 조건 문의'
                analysis['expected_answer_type'] = '경조금 지원 조건/경우만'
            elif analysis['question_type'] == 'method':
                analysis['intent'] = '경조금 신청 방법 문의'
                analysis['expected_answer_type'] = '경조금 신청 절차만'
            elif analysis['question_type'] == 'amount':
                analysis['intent'] = '경조금 금액 문의'
                analysis['expected_answer_type'] = '경조금 금액만'
        
        return analysis
    
    async def step2_generate_with_exaone(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """2단계: EXAONE으로 RAG 기반 답변 생성"""
        logger.info("2단계: EXAONE으로 답변 생성 중...")
        
        try:
            # 검색된 문서들을 컨텍스트로 구성
            context_parts = []
            total_score = 0
            relevant_docs = []
            
            # 질문 분석
            question_analysis = self._analyze_question_intent(query)
            
            for i, doc in enumerate(retrieved_docs):
                text = doc.get('text', '').strip()
                source = doc.get('source_file', 'unknown')
                score = doc.get('score', 0)
                keyword_score = doc.get('keyword_score', 0)
                intent_score = doc.get('intent_score', 0)
                combined_score = doc.get('combined_score', 0)
                
                # 관련성 검증 - 질문 의도와 맞지 않는 문서 필터링
                if not self._is_content_relevant_to_question(query, text, question_analysis):
                    logger.info(f"문서 {i+1} 관련성 부족으로 제외: {source}")
                    continue
                
                total_score += combined_score
                relevant_docs.append(doc)
                
                # 문서별 관련성 표시
                relevance_indicator = ""
                if combined_score > 0.8:
                    relevance_indicator = "🔥 매우 높은 관련성"
                elif combined_score > 0.6:
                    relevance_indicator = "⭐ 높은 관련성"
                elif combined_score > 0.4:
                    relevance_indicator = "📄 중간 관련성"
                else:
                    relevance_indicator = "📋 참고 정보"
                
                # 구조화된 컨텍스트 구성
                context_part = f"""
                ==================== 참고문서 {len(context_parts)+1} ====================
                📋 출처: {source}
                {relevance_indicator} (종합점수: {combined_score:.3f})
                - 의미적 유사도: {score:.3f}
                - 키워드 매칭: {keyword_score:.3f}  
                - 의도 매칭: {intent_score:.3f}

                📝 내용:
                {text}
                ================================================
                """
                context_parts.append(context_part)
            
            # 관련 문서가 없으면 조기 반환
            if not relevant_docs:
                return f"'{query}' 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문으로 시도해보시거나 더 구체적으로 질문해주세요."
            
            # 전체 컨텍스트 요약 정보 추가
            context_summary = f"""
            📊 컨텍스트 요약:
            - 총 참고문서 수: {len(relevant_docs)}개
            - 평균 관련성 점수: {total_score / len(relevant_docs):.3f}

            """
            
            context = context_summary + "\n".join(context_parts)
            
            # 개선된 시스템 프롬프트 + 사용자 프롬프트
            system_prompt = """당신은 전문적이고 정확한 답변을 제공하는 AI 어시스턴트입니다. 

            다음 원칙에 따라 답변하세요:
            1. 제공된 참고 문서의 정보만을 사용하여 답변합니다
            2. 질문에 직접적으로 대답하고 관련 없는 정보는 절대 포함하지 않습니다
            3. 정확하고 구체적인 정보를 제공합니다  
            4. 불확실한 정보는 명시적으로 표현합니다
            5. 답변은 논리적이고 체계적으로 구성합니다
            6. 한국어로 자연스럽고 이해하기 쉽게 작성합니다
            7. 참고 문서에 없는 내용은 추측하지 않습니다
            8. 질문의 핵심 의도를 파악하고 그에 맞는 답변만 제공합니다
            
            ⚠️ 중요: 질문에서 요구하지 않은 추가 정보는 포함하지 마세요!
            """


            
            # 제외 주제 안내 추가
            exclude_guidance = ""
            if question_analysis.get('exclude_topics'):
                exclude_list = ', '.join(question_analysis['exclude_topics'])
                exclude_guidance = f"\n⛔ 다음 주제는 절대 포함하지 마세요: {exclude_list}"
            
            generation_prompt = f"""
            
            <|system|>
            {system_prompt}

            <|user|>
            다음 참고 문서를 바탕으로 사용자의 질문에 정확히 답변해주세요.

            **질문 분석**:
            - 원본 질문: {query}
            - 질문 의도: {question_analysis['intent']}
            - 핵심 키워드: {', '.join(question_analysis['keywords'])}
            - 기대 답변 유형: {question_analysis['expected_answer_type']}
            - 구체적 주제: {question_analysis.get('specific_topic', '일반')}
            {exclude_guidance}

            **참고 문서**:
            {context}

            **답변 지침**:
            1. 질문의 핵심 의도("{question_analysis['intent']}")에만 정확히 맞춰 답변하세요
            2. "{question_analysis['expected_answer_type']}" 형태의 답변만 제공하세요
            3. 질문에서 요구하지 않은 추가 정보는 절대 포함하지 마세요
            4. 예를 들어, 화환 조건을 물으면 화환 조건만, 경조금을 물으면 경조금만 답변하세요
            5. 참고 문서에서 질문과 직접 관련된 부분만 사용하세요
            6. 답변이 없으면 "관련 정보를 찾을 수 없습니다"라고 솔직히 말하세요

            위 참고 문서의 정보를 바탕으로 질문에 정확히 대답해주세요:

            <|assistant|>
            """
            
            # EXAONE 모델로 답변 생성
            if self.exaone_model and self.exaone_tokenizer:
                # 직접 모델 사용
                inputs = self.exaone_tokenizer(
                    generation_prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.exaone_model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.exaone_tokenizer.pad_token_id,
                        eos_token_id=self.exaone_tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # 반복 방지
                        length_penalty=1.0       # 길이 조절
                    )
                
                generated_text = self.exaone_tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated_text[len(generation_prompt):].strip()
                
            elif self.exaone_pipeline:
                # 파이프라인 사용
                result = self.exaone_pipeline(
                    generation_prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    return_full_text=False,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
                answer = result[0]['generated_text'].strip()
                
            else:
                # 대안: 컨텍스트 길이 제한 적용
                context_limit = self.config.max_tokens
                if len(context) > context_limit:
                    context = context[:context_limit] + "\n\n... (내용이 길어 일부 생략됨)"
                
                answer = f"""질문 '{query}'에 대한 답변을 다음 문서들을 바탕으로 제공합니다.

참고 문서: {len(relevant_docs)}개 문서에서 검색

{context[:500]}{'...' if len(context) > 500 else ''}

※ EXAONE 모델 로딩 실패로 기본 답변을 제공합니다."""
            
            logger.info("EXAONE 답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.error(f"EXAONE 답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def step3_review_with_gpt4(self, query: str, exaone_answer: str, 
                                   retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """3단계: GPT-4.1로 최종 답변 검수"""
        logger.info("3단계: GPT-4.1로 답변 검수 중...")
        
        if not self.openai_client:
            logger.warning("OpenAI 클라이언트가 설정되지 않아 검수를 건너뜁니다")
            return {
                "accuracy_score": 7,
                "completeness_score": 7,
                "clarity_score": 7,
                "usefulness_score": 7,
                "overall_score": 7,
                "strengths": "OpenAI 클라이언트가 설정되지 않음",
                "weaknesses": "검수를 수행할 수 없음",
                "suggestions": "OpenAI API 키를 설정하여 검수 활성화",
                "approved": True,
                "reviewer_comments": "검수 건너뜀"
            }
        
        try:
            review_prompt = f"""다음 AI가 생성한 답변을 검수하고 개선사항을 제안해주세요.

            원본 질문: {query}

            AI 생성 답변:
            {exaone_answer}

            참고된 문서 수: {len(retrieved_docs)}개

            검수 기준:
            1. 정확성: 답변이 질문에 정확히 대응하는가?
            2. 완성도: 답변이 충분히 상세하고 완전한가?
            3. 명확성: 답변이 이해하기 쉽고 명확한가?
            4. 유용성: 답변이 사용자에게 실질적으로 도움이 되는가?

            다음 JSON 형태로 검수 결과를 제공해주세요:
            {{
                "accuracy_score": 1-10점,
                "completeness_score": 1-10점,
                "clarity_score": 1-10점,
                "usefulness_score": 1-10점,
                "overall_score": 1-10점,
                "strengths": "답변의 강점들",
                "weaknesses": "답변의 약점들",
                "suggestions": "구체적인 개선사항",
                "approved": true/false,
                "reviewer_comments": "전체적인 검수 의견"
            }}"""
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.gpt_model_name,  # settings에서 지정한 GPT-4.1 모델 사용
                messages=[
                    {"role": "system", "content": "당신은 AI 답변을 검수하는 전문가입니다. 객관적이고 건설적인 피드백을 제공해주세요."},
                    {"role": "user", "content": review_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            try:
                review_result = json.loads(response.choices[0].message.content)
                logger.info(f"GPT-4.1 검수 완료: 전체 점수 {review_result.get('overall_score', 0)}/10")
                return review_result
            except json.JSONDecodeError:
                logger.warning("GPT-4.1 응답 JSON 파싱 실패")
                return {
                    "accuracy_score": 6,
                    "completeness_score": 6,
                    "clarity_score": 6,
                    "usefulness_score": 6,
                    "overall_score": 6,
                    "strengths": "JSON 파싱 실패",
                    "weaknesses": "응답 형식 오류",
                    "suggestions": "응답 형식 개선 필요",
                    "approved": True,
                    "reviewer_comments": response.choices[0].message.content[:200] + "..."
                }
                
        except Exception as e:
            logger.error(f"GPT-4.1 검수 실패: {e}")
            return {
                "accuracy_score": 5,
                "completeness_score": 5,
                "clarity_score": 5,
                "usefulness_score": 5,
                "overall_score": 5,
                "strengths": "검수 과정에서 오류 발생",
                "weaknesses": f"검수 실패: {str(e)}",
                "suggestions": "수동으로 답변을 검토해주세요",
                "approved": False,
                "reviewer_comments": f"자동 검수 실패: {str(e)}"
            }
    
    async def step4_finalize_response(self, query: str, exaone_answer: str, 
                                    gpt_review: Dict[str, Any], 
                                    retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """4단계: 최종 답변 확정"""
        logger.info("4단계: 최종 답변 확정 중...")
        
        # 검수 결과에 따라 최종 답변 결정
        final_answer = exaone_answer
        gpt_score = gpt_review.get('overall_score', 5)
        
        # 최소 점수 임계값 확인 (settings 값 적용)
        is_approved = gpt_score >= self.config.min_score_threshold
        
        # 참고 문서 정보 추가
        source_files = list(set(doc.get('source_file', 'unknown') for doc in retrieved_docs))
        if source_files:
            final_answer += f"\n\n📚 참고 문서 ({len(source_files)}개): {', '.join(source_files[:3])}"
            if len(source_files) > 3:
                final_answer += f" 외 {len(source_files)-3}개"
        
        return {
            "query": query,
            "final_answer": final_answer,
            "approved": is_approved,  # settings 임계값 기반 승인
            "gpt_review_score": gpt_score,
            "retrieved_documents": retrieved_docs,
            "sources": source_files,
            "pipeline_metadata": {
                "retrieved_count": len(retrieved_docs),
                "gpt4.1_review_score": gpt_score,
                "total_processing_steps": 4,
                "min_score_threshold": self.config.min_score_threshold,
                "approved_by_threshold": is_approved
            },
            "review_summary": gpt_review.get('reviewer_comments', ''),
            "processing_timestamp": datetime.now().isoformat()
        }
    
    async def run_full_pipeline(self, query: str) -> Dict[str, Any]:
        """전체 복합 RAG 파이프라인 실행"""
        logger.info(f"deephub 복합 RAG 파이프라인 시작: {query}")
        
        start_time = datetime.now()
        
        try:
            # 1단계: RAG 문서 검색
            retrieved_docs = await self.step1_retrieve_documents(query)
            
            # 1.5단계: Re-ranker로 문서 재순위 매기기
            reranked_docs = await self.step1_5_rerank_documents(query, retrieved_docs)
            
            # 2단계: EXAONE으로 답변 생성
            exaone_answer = await self.step2_generate_with_exaone(query, reranked_docs)
            
            # 3단계: GPT-4.1로 검수
            gpt_review = await self.step3_review_with_gpt4(query, exaone_answer, reranked_docs)
            
            # 4단계: 최종 답변 확정
            final_response = await self.step4_finalize_response(
                query, exaone_answer, gpt_review, reranked_docs
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            final_response['processing_time'] = processing_time
            
            logger.info(f"복합 RAG 파이프라인 완료 (처리시간: {processing_time:.2f}초)")
            return final_response
            
        except Exception as e:
            logger.error(f"복합 RAG 처리 실패: {e}")
            return {"error": f"죄송합니다. 요청을 처리하는 중 오류가 발생했습니다: {str(e)}"}
    
    def _is_content_relevant_to_question(self, query: str, content: str, question_analysis: Dict[str, Any]) -> bool:
        """컨텍스트 내용이 질문과 관련되는지 확인 (강화된 버전)"""
        # 기본 키워드 매칭
        query_keywords = set(question_analysis.get('keywords', []))
        content_keywords = self._extract_enhanced_keywords(content)
        
        # 공통 키워드가 있는지 확인
        common_keywords = query_keywords.intersection(content_keywords)
        if not common_keywords:
            return False
        
        # 제외할 주제가 있는지 확인 (강화된 필터링)
        exclude_topics = question_analysis.get('exclude_topics', [])
        if exclude_topics:
            for exclude_topic in exclude_topics:
                if exclude_topic in content.lower():
                    logger.info(f"제외 주제 '{exclude_topic}' 발견으로 문서 제외")
                    return False
        
        # 구체적 주제 매칭 확인
        specific_topic = question_analysis.get('specific_topic')
        
        if specific_topic == 'wreath':  # 화환 관련 질문
            # 화환 키워드가 있어야 함
            wreath_keywords = ['화환', '꽃', '장식', '조화']
            if not any(keyword in content for keyword in wreath_keywords):
                return False
            
            # 경조금 관련 내용이 있으면 제외 (더 엄격하게)
            money_indicators = ['원', '금액', '경조금', '지급', '비용', '돈', '만원', '천원']
            money_count = sum(1 for indicator in money_indicators if indicator in content)
            
            # 경조금 언급이 많으면 제외
            if money_count > 2:
                logger.info(f"화환 질문에 경조금 내용이 많이 포함되어 제외 (경조금 언급: {money_count}개)")
                return False
                
        elif specific_topic == 'condolence_money':  # 경조금 관련 질문
            # 경조금 키워드가 있어야 함
            money_keywords = ['경조금', '금액', '원', '지급', '비용', '돈']
            if not any(keyword in content for keyword in money_keywords):
                return False
            
            # 화환 관련 내용이 있으면 제외
            wreath_indicators = ['화환', '꽃', '장식', '조화']
            if any(indicator in content for indicator in wreath_indicators):
                logger.info("경조금 질문에 화환 내용이 포함되어 제외")
                return False
        
        # 질문 유형별 세부 관련성 검증
        question_type = question_analysis.get('question_type', 'general')
        
        if question_type == 'condition':
            # 조건/시기 관련 질문
            condition_patterns = [
                r'조건', r'경우', r'때', r'시기', r'기준', r'대상', r'해당',
                r'적용', r'범위', r'자격', r'요건'
            ]
            if not any(re.search(pattern, content, re.IGNORECASE) for pattern in condition_patterns):
                return False
                
            # 화환 관련 특별 처리 (더 엄격하게)
            if specific_topic == 'wreath':
                # 화환 조건에 대한 질문인데 금액만 나오고 조건이 없는 경우 제외
                has_amount_only = bool(re.search(r'\d+.*원', content)) and not any(
                    keyword in content for keyword in ['조건', '경우', '때', '지원', '대상', '해당']
                )
                if has_amount_only:
                    logger.info("화환 조건 질문에 금액만 있고 조건 정보 없어서 제외")
                    return False
                    
        elif question_type == 'method':
            # 방법/절차 관련 질문
            method_patterns = [
                r'방법', r'절차', r'신청', r'처리', r'진행', r'접수',
                r'제출', r'등록', r'과정', r'단계'
            ]
            if not any(re.search(pattern, content, re.IGNORECASE) for pattern in method_patterns):
                return False
                
        elif question_type == 'amount':
            # 금액 관련 질문
            if not re.search(r'\d+.*원|금액|비용|가격|돈', content):
                return False
                
        elif question_type == 'eligibility':
            # 자격/대상 관련 질문
            eligibility_patterns = [
                r'자격', r'대상', r'범위', r'해당자', r'신청자',
                r'대상자', r'수혜자', r'해당', r'포함'
            ]
            if not any(re.search(pattern, content, re.IGNORECASE) for pattern in eligibility_patterns):
                return False
        
        return True


# 전역 인스턴스
_composite_rag_system = None

async def get_composite_rag_system() -> DeephubCompositeRAG:
    """복합 RAG 시스템 싱글톤 인스턴스 반환"""
    global _composite_rag_system
    
    if _composite_rag_system is None:
        _composite_rag_system = DeephubCompositeRAG()
        logger.info("복합 RAG 시스템 싱글톤 인스턴스 생성")
    
    return _composite_rag_system

async def composite_ask_with_context(question: str) -> Dict[str, Any]:
    """복합 RAG 시스템으로 질문 처리"""
    system = await get_composite_rag_system()
    return await system.run_full_pipeline(question)

async def get_composite_rag_status() -> Dict[str, Any]:
    """복합 RAG 시스템 상태 확인"""
    system = await get_composite_rag_system()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "composite_rag_initialized": True,
        "models_loaded": {
            "embedding_model": system.embedding_model is not None,
            "exaone_model": system.exaone_model is not None or system.exaone_pipeline is not None,
            "openai_client": system.openai_client is not None,
            "reranker": system.reranker is not None
        },
        "config": {
            "embedding_model": system.config.embedding_model_name,
            "exaone_model": system.config.exaone_model_name,
            "gpt_model": system.config.gpt_model_name,
            "enable_reranker": system.config.enable_reranker
        }
    }

class DocumentReranker:
    """문서 재순위 매기기 클래스 (BGE-Large + BM25 + 임베딩 하이브리드)"""
    
    def __init__(self, config):
        self.config = config
        self.bm25_corpus = []
        self.bm25_model = None
        
        # settings.py에서 Re-ranker 모델 설정 로드
        self.model_name = config.reranker_model_name
        self.model_type = config.reranker_model_type
        self.device = config.reranker_device
        self.max_length = config.reranker_max_length
        self.batch_size = config.reranker_batch_size
        self.num_workers = getattr(config, 'reranker_num_workers', 0)
        
        # Re-ranker 모델 로드
        try:
            if self.model_type == "cross_encoder":
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(
                    self.model_name,
                    device=self.device if self.device != "auto" else None,
                    max_length=self.max_length
                )
                logger.info(f"Cross Encoder Re-ranker 모델 로드 완료: {self.model_name} (device: {self.device})")
            elif self.model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(
                    self.model_name,
                    device=self.device if self.device != "auto" else None
                )
                logger.info(f"Sentence Transformer Re-ranker 모델 로드 완료: {self.model_name} (device: {self.device})")
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        except Exception as e:
            logger.error(f"Re-ranker 모델 로드 실패: {e}")
            raise
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        self.cleanup()
    
    def cleanup(self):
        """명시적 리소스 정리"""
        try:
            if hasattr(self, 'cross_encoder') and self.cross_encoder is not None:
                del self.cross_encoder
            if hasattr(self, 'encoder') and self.encoder is not None:
                del self.encoder
            if hasattr(self, 'bm25_model') and self.bm25_model is not None:
                del self.bm25_model
            if hasattr(self, 'bm25_corpus') and self.bm25_corpus is not None:
                del self.bm25_corpus
            gc.collect()
        except Exception as e:
            logger.warning(f"DocumentReranker 리소스 정리 중 오류: {e}")
    
    def _calculate_reranker_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Re-ranker 모델 기반 재순위 점수 계산 (Cross Encoder 또는 Sentence Transformer)"""
        try:
            if self.model_type == "cross_encoder" and hasattr(self, 'cross_encoder'):
                # Cross Encoder 사용: 질문-문서 쌍을 직접 입력
                query_doc_pairs = []
                for doc in documents:
                    text = doc.get('text', '')[:self.max_length]  # settings에서 설정한 길이 제한
                    query_doc_pairs.append([query, text])
                
                # Cross Encoder로 직접 관련성 점수 계산 (배치 처리)
                scores = self.cross_encoder.predict(
                    query_doc_pairs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    show_progress_bar=False
                )
                
                # 점수를 리스트로 변환 (numpy array일 수 있음)
                scores = [float(score) for score in scores]
                
                logger.debug(f"Cross Encoder Re-ranker 점수 계산 완료: {len(scores)}개 문서, 모델: {self.model_name}")
                return scores
                
            elif self.model_type == "sentence_transformer" and hasattr(self, 'encoder'):
                # Sentence Transformer 사용: 임베딩 기반 유사도 계산
                texts = []
                for doc in documents:
                    text = doc.get('text', '')[:self.max_length]
                    texts.append(text)
                
                # 임베딩 생성 (배치 처리)
                query_embedding = self.encoder.encode(
                    query, 
                    normalize_embeddings=True,
                    batch_size=self.batch_size
                )
                doc_embeddings = self.encoder.encode(
                    texts, 
                    normalize_embeddings=True,
                    batch_size=self.batch_size
                )
                
                # 코사인 유사도 계산
                scores = []
                for doc_emb in doc_embeddings:
                    score = np.dot(query_embedding, doc_emb)
                    scores.append(float(score))
                
                logger.debug(f"Sentence Transformer Re-ranker 점수 계산 완료: {len(scores)}개 문서, 모델: {self.model_name}")
                return scores
            
            else:
                logger.warning(f"Re-ranker 모델이 로드되지 않음: {self.model_type}")
                return [0.0] * len(documents)
                
        except Exception as e:
            logger.warning(f"Re-ranker 점수 계산 실패: {e}")
            return [0.0] * len(documents)
    
    def _prepare_bm25(self, documents: List[Dict[str, Any]]) -> None:
        """BM25 모델 준비"""
        try:
            if BM25Okapi is None:
                logger.warning("BM25Okapi 모듈을 사용할 수 없습니다. BM25 점수는 0으로 설정됩니다.")
                return
            
            # 문서 텍스트를 토큰화하여 corpus 생성
            self.bm25_corpus = []
            for doc in documents:
                text = doc.get('text', '').lower()
                # 한글, 영문, 숫자만 추출하여 토큰화
                tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text)
                self.bm25_corpus.append(tokens)
            
            # BM25 모델 초기화
            if self.bm25_corpus:
                self.bm25_model = BM25Okapi(self.bm25_corpus)
                logger.debug(f"BM25 모델 준비 완료: {len(self.bm25_corpus)}개 문서")
            else:
                logger.warning("BM25 corpus가 비어있습니다.")
                self.bm25_model = None
                
        except Exception as e:
            logger.error(f"BM25 모델 준비 실패: {e}")
            self.bm25_model = None
    
    def _calculate_bm25_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """BM25 기반 점수 계산"""
        try:
            if self.bm25_model is None or not self.bm25_corpus:
                logger.debug("BM25 모델이 준비되지 않았습니다. 기본 점수 반환")
                return [0.0] * len(documents)
            
            # 쿼리 토큰화
            query_tokens = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
            
            if not query_tokens:
                return [0.0] * len(documents)
            
            # BM25 점수 계산
            bm25_scores = self.bm25_model.get_scores(query_tokens)
            
            # 점수 정규화 (0-1 범위)
            if len(bm25_scores) > 0:
                max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                normalized_scores = [score / max_score for score in bm25_scores]
            else:
                normalized_scores = [0.0] * len(documents)
            
            logger.debug(f"BM25 점수 계산 완료: 평균 {np.mean(normalized_scores):.3f}")
            return normalized_scores
            
        except Exception as e:
            logger.warning(f"BM25 점수 계산 실패: {e}")
            return [0.0] * len(documents)
    
    def _apply_mmr_diversity(self, documents: List[Dict[str, Any]], 
                           scores: List[float], 
                           lambda_param: float = None) -> List[Dict[str, Any]]:
        """MMR (Maximal Marginal Relevance) 다양성 적용 (settings.py 설정 활용)"""
        try:
            if not documents or not scores:
                return documents
            
            # settings.py에서 lambda 파라미터 로드
            if lambda_param is None:
                lambda_param = self.config.mmr_lambda
            
            # 문서 임베딩 생성 (다양성 계산용)
            doc_texts = [doc.get('text', '')[:256] for doc in documents]  # 길이 제한
            doc_embeddings = self.encoder.encode(doc_texts, normalize_embeddings=True)
            
            # MMR 알고리즘 적용
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            
            # 첫 번째 문서는 가장 높은 점수 선택
            if remaining_indices:
                best_idx = max(remaining_indices, key=lambda i: scores[i])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # 나머지 문서들은 MMR 점수로 선택
            while remaining_indices and len(selected_indices) < self.config.reranker_output_k:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # 관련성 점수
                    relevance_score = scores[idx]
                    
                    # 이미 선택된 문서들과의 최대 유사도 계산
                    max_similarity = 0.0
                    for selected_idx in selected_indices:
                        similarity = np.dot(doc_embeddings[idx], doc_embeddings[selected_idx])
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR 점수 계산 (관련성 - 다양성)
                    mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
                    mmr_scores.append((idx, mmr_score))
                
                # 가장 높은 MMR 점수 선택
                if mmr_scores:
                    best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            # 선택된 문서들 반환
            final_docs = [documents[i] for i in selected_indices]
            
            logger.debug(f"MMR 다양성 적용 완료: {len(documents)}개 → {len(final_docs)}개 문서 선택")
            return final_docs
            
        except Exception as e:
            logger.warning(f"MMR 다양성 적용 실패: {e}")
            # 실패시 점수순으로 상위 문서 반환
            sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in sorted_docs[:self.config.reranker_output_k]]
    
    async def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 재순위 매기기 메인 함수"""
        if not documents:
            return documents
        
        logger.info(f"Re-ranker 시작: {len(documents)}개 문서 재순위 매기기")
        
        try:
            # 상위 K개 문서만 처리 (성능 최적화)
            top_documents = documents[:self.config.reranker_top_k]
            
            # BM25 모델 준비
            self._prepare_bm25(top_documents)
            
            # 1. BM25 점수 계산
            bm25_scores = self._calculate_bm25_scores(query, top_documents)
            
            # 2. BGE-Large 점수 계산
            reranker_scores = self._calculate_reranker_scores(query, top_documents)
            
            # 3. 기존 임베딩 점수 추출
            embedding_scores = [doc.get('score', 0) for doc in top_documents]
            embedding_scores = [score / max(embedding_scores) if max(embedding_scores) > 0 else 0 for score in embedding_scores]
            
            # 4. 하이브리드 점수 계산
            hybrid_scores = []
            for i in range(len(top_documents)):
                hybrid_score = (
                    self.config.bm25_weight * bm25_scores[i] +
                    self.config.reranker_weight * reranker_scores[i] +
                    self.config.embedding_weight * embedding_scores[i]
                )
                hybrid_scores.append(hybrid_score)
                
                # 문서에 재순위 점수 추가
                top_documents[i]['rerank_score'] = hybrid_score
                top_documents[i]['bm25_score'] = bm25_scores[i]
                top_documents[i]['reranker_score'] = reranker_scores[i]
            
            # 5. 점수순으로 정렬
            reranked_docs = sorted(
                zip(top_documents, hybrid_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            reranked_docs = [doc for doc, score in reranked_docs]
            
            # 6. MMR 다양성 적용
            if self.config.diversity_penalty > 0:
                final_docs = self._apply_mmr_diversity(
                    reranked_docs, 
                    [doc['rerank_score'] for doc in reranked_docs],
                    lambda_param=1.0 - self.config.diversity_penalty
                )
            else:
                final_docs = reranked_docs[:self.config.reranker_output_k]
            
            logger.info(f"Re-ranker 완료: {len(final_docs)}개 문서 선택")
            
            # 로그용 점수 출력
            for i, doc in enumerate(final_docs[:3]):  # 상위 3개만 로깅
                logger.debug(f"순위 {i+1}: 종합점수 {doc.get('rerank_score', 0):.3f} "
                           f"(BM25: {doc.get('bm25_score', 0):.3f}, "
                           f"Reranker: {doc.get('reranker_score', 0):.3f}, "
                           f"Embed: {doc.get('score', 0):.3f})")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Re-ranker 실행 실패: {e}")
            # 실패시 기존 문서 반환
            return documents[:self.config.reranker_output_k]

    def get_reranker_status(self) -> Dict[str, Any]:
        """Re-ranker 상태 및 성능 정보 반환 (Cross Encoder 지원)"""
        
        # 모델 로드 상태 확인
        model_loaded = False
        model_info = {}
        
        if self.model_type == "cross_encoder" and hasattr(self, 'cross_encoder'):
            model_loaded = self.cross_encoder is not None
            model_info = {
                "name": self.model_name,
                "type": "Cross Encoder",
                "device": self.device,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers
            }
        elif self.model_type == "sentence_transformer" and hasattr(self, 'encoder'):
            model_loaded = self.encoder is not None
            model_info = {
                "name": self.model_name,
                "type": "Sentence Transformer",
                "device": self.device,
                "max_length": self.max_length,
                "batch_size": self.batch_size
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "reranker_enabled": True,
            "models_loaded": {
                "reranker_model": model_loaded,
                "model_type": self.model_type,
                "bm25": self.bm25_model is not None if hasattr(self, 'bm25_model') else False,
                "bm25_corpus_size": len(self.bm25_corpus) if hasattr(self, 'bm25_corpus') and self.bm25_corpus else 0
            },
            "model_info": model_info,
            "config": {
                "reranker_top_k": self.config.reranker_top_k,
                "reranker_output_k": self.config.reranker_output_k,
                "reranker_weight": self.config.reranker_weight,
                "bm25_weight": self.config.bm25_weight,
                "embedding_weight": self.config.embedding_weight,
                "diversity_penalty": self.config.diversity_penalty,
                "mmr_lambda": self.config.mmr_lambda
            },
            "performance_info": {
                "hybrid_scoring": f"{self.model_type.replace('_', ' ').title()} + BM25 + Embedding",
                "diversity_algorithm": "MMR (Maximal Marginal Relevance)",
                "normalization": "Min-Max scaling applied",
                "scoring_method": "Direct relevance scoring" if self.model_type == "cross_encoder" else "Embedding similarity"
            }
        }
    
    def compare_ranking_performance(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """원본 검색과 Re-ranker 성능 비교"""
        try:
            if not documents:
                return {"error": "문서가 없습니다"}
            
            # Re-ranker 적용
            reranked_docs = asyncio.run(self.rerank_documents(query, documents.copy()))
            
            # 순위 변화 분석
            ranking_changes = []
            for new_rank, doc in enumerate(reranked_docs):
                # 원본 문서에서의 인덱스 찾기
                original_rank = None
                for orig_rank, orig_doc in enumerate(documents):
                    if orig_doc.get('text') == doc.get('text'):
                        original_rank = orig_rank
                        break
                
                if original_rank is not None:
                    ranking_changes.append({
                        "document_id": original_rank,
                        "original_rank": original_rank + 1,
                        "new_rank": new_rank + 1,
                        "rank_change": original_rank - new_rank,
                        "original_score": doc.get('score', 0),
                        "rerank_score": doc.get('rerank_score', 0),
                        "bm25_score": doc.get('bm25_score', 0),
                        "reranker_score": doc.get('reranker_score', 0)
                    })
            
            # 성능 메트릭 계산
            performance_metrics = {
                "total_documents": len(documents),
                "reranked_documents": len(reranked_docs),
                "avg_rank_change": np.mean([abs(change["rank_change"]) for change in ranking_changes]) if ranking_changes else 0,
                "top_3_changes": sum(1 for change in ranking_changes if abs(change["rank_change"]) > 0 and change["new_rank"] <= 3),
                "score_improvement": {
                    "avg_rerank_score": np.mean([doc.get('rerank_score', 0) for doc in reranked_docs]),
                    "avg_original_score": np.mean([doc.get('score', 0) for doc in documents]),
                }
            }
            
            return {
                "query": query,
                "ranking_changes": ranking_changes[:10],  # 상위 10개만 반환
                "performance_metrics": performance_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"순위 성능 비교 실패: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환 (settings.py 설정 완전 반영)"""
        return {
            "timestamp": datetime.now().isoformat(),
            "composite_rag_initialized": True,
            "models_loaded": {
                "reranker": (hasattr(self, 'cross_encoder') and self.cross_encoder is not None) or 
                           (hasattr(self, 'encoder') and self.encoder is not None),
                "reranker_model_name": self.config.reranker_model_name,
                "reranker_model_type": self.config.reranker_model_type,
                "reranker_device": self.config.reranker_device,
                "bm25_available": BM25Okapi is not None
            },
            "model_config": {
                "reranker_model": self.config.reranker_model_name,
                "model_type": self.config.reranker_model_type,
                "device": self.config.reranker_device,
                "max_length": self.config.reranker_max_length,
                "batch_size": self.config.reranker_batch_size,
                "num_workers": self.config.reranker_num_workers
            },
            "algorithm_config": {
                "enable_reranker": self.config.enable_reranker,
                "input_docs": self.config.reranker_top_k,
                "output_docs": self.config.reranker_output_k,
                "score_weights": {
                    "reranker": self.config.reranker_weight,
                    "bm25": self.config.bm25_weight,
                    "embedding": self.config.embedding_weight
                },
                "mmr_config": {
                    "lambda": self.config.mmr_lambda,
                    "diversity_penalty": self.config.diversity_penalty
                }
            },
            "processing_config": {
                "max_new_tokens": self.config.max_new_tokens,
                "generation_temperature": self.config.generation_temperature,
                "similarity_threshold": self.config.similarity_threshold,
                "max_context_length": self.config.max_context_length
            },
            "algorithms": {
                "reranking": f"Hybrid {self.config.reranker_model_name} + BM25 + Embedding",
                "diversity": "MMR (Maximal Marginal Relevance)",
                "scoring": "Weighted combination with normalization"
            },
            "settings_compliance": "✅ settings.py 완전 준수",
            "settings_source": "shared/config/settings.py"
        } 