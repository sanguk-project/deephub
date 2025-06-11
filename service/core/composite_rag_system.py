"""
deephub 복합 RAG 시스템
사용자 프롬프트 → RAG → EXAONE 3.5 답변생성 → GPT-4 검수 → 최종답변
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# deephub 내부 컴포넌트 import
from service.storage.vector_store import search_similar_documents
# MongoDB 로깅 제거됨
from shared.utils import get_openai_client
from shared.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CompositeRAGConfig:
    """복합 RAG 시스템 설정"""
    # 모델 설정 (settings에서 자동 로드)
    embedding_model_name: str = settings.model.embedding_model
    exaone_model_name: str = settings.model.exaone_model_path
    gpt_model_name: str = settings.model.final_answer_model
    
    # RAG 설정
    top_k_documents: int = settings.rag.max_retrieved_docs
    similarity_threshold: float = settings.rag.similarity_threshold
    
    # 생성 설정
    max_new_tokens: int = 1024
    temperature: float = 0.7


class DeephubCompositeRAG:
    """deephub 복합 RAG 시스템"""
    
    def __init__(self, config: CompositeRAGConfig = None):
        # 기본 설정 로드
        self.config = config or CompositeRAGConfig()
        
        # OpenAI 클라이언트 설정 (공용 유틸리티 사용)
        try:
            self.openai_client = get_openai_client(async_mode=True)
        except ValueError as e:
            logger.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
        
        # 임베딩 모델 로드
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model_name
        )
        
        # EXAONE 모델 로드
        self._load_exaone_model()
        
        # 로깅 기능 제거됨
        
        logger.info("deephub 복합 RAG 시스템 초기화 완료")
    
    def _load_exaone_model(self):
        """EXAONE 모델 로드"""
        try:
            logger.info(f"EXAONE 모델 로딩 중: {self.config.exaone_model_name}")
            
            self.exaone_tokenizer = AutoTokenizer.from_pretrained(
                self.config.exaone_model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.exaone_tokenizer.pad_token is None:
                self.exaone_tokenizer.pad_token = self.exaone_tokenizer.eos_token
            
            self.exaone_model = AutoModelForCausalLM.from_pretrained(
                self.config.exaone_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.exaone_pipeline = None
            logger.info("EXAONE 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"EXAONE 모델 로딩 실패: {e}")
            logger.info("파이프라인 모드로 대안 설정 중...")
            
            try:
                self.exaone_pipeline = pipeline(
                    "text-generation",
                    model=self.config.exaone_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.exaone_model = None
                self.exaone_tokenizer = None
                logger.info("EXAONE 파이프라인 설정 완료")
            except Exception as e2:
                logger.error(f"EXAONE 파이프라인 설정 실패: {e2}")
                self.exaone_pipeline = None
                self.exaone_model = None
                self.exaone_tokenizer = None
    
    async def step1_retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """1단계: RAG 문서 검색 (향상된 검색 로직)"""
        logger.info("1단계: RAG 문서 검색 중...")
        
        try:
            # 1차 검색: 임베딩 벡터 생성
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # 2차 검색: 다양한 검색 전략 적용
            # 더 많은 문서를 1차로 가져온 후 필터링
            initial_docs = await search_similar_documents(
                query_embedding=query_embedding,
                limit=self.config.top_k_documents * 3  # 3배수로 검색
            )
            
            # 문서 품질 평가 및 필터링
            filtered_docs = []
            
            for doc in initial_docs:
                # 유사도 임계값 확인
                if doc.get('score', 0) < self.config.similarity_threshold:
                    continue
                
                # 문서 내용 품질 확인
                text = doc.get('text', '').strip()
                if len(text) < 20:  # 너무 짧은 내용 제외
                    continue
                
                # 키워드 관련성 검사
                query_keywords = self._extract_keywords(query)
                doc_keywords = self._extract_keywords(text.lower())
                
                # 키워드 매칭 점수 계산
                keyword_score = self._calculate_keyword_overlap(query_keywords, doc_keywords)
                doc['keyword_score'] = keyword_score
                
                filtered_docs.append(doc)
            
            # 다중 기준으로 정렬 (유사도 + 키워드 점수)
            filtered_docs.sort(key=lambda x: (x.get('score', 0) * 0.7 + x.get('keyword_score', 0) * 0.3), reverse=True)
            
            # 중복 제거 - 같은 source_file에서 너무 많은 문서 방지
            deduplicated_docs = self._deduplicate_and_diversify(filtered_docs)
            
            # 최종 문서 수 제한
            final_docs = deduplicated_docs[:self.config.top_k_documents]
            
            logger.info(f"검색 완료: {len(final_docs)}개 문서 발견 (초기: {len(initial_docs)}개)")
            return final_docs
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> set:
        """텍스트에서 키워드 추출 (간단한 버전)"""
        import re
        
        # 한글, 영문, 숫자만 추출하고 불용어 제거
        stopwords = {'이', '그', '저', '것', '들', '는', '은', '을', '를', '의', '에', '와', '과', 
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        keywords = {word for word in words if len(word) > 1 and word not in stopwords}
        
        return keywords
    
    def _calculate_keyword_overlap(self, query_keywords: set, doc_keywords: set) -> float:
        """키워드 중복도 계산"""
        if not query_keywords:
            return 0.0
        
        overlap = len(query_keywords.intersection(doc_keywords))
        return overlap / len(query_keywords)
    
    def _deduplicate_and_diversify(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 및 다양성 확보"""
        source_count = {}
        result = []
        
        for doc in docs:
            source = doc.get('source_file', 'unknown')
            
            # 같은 소스에서 최대 3개까지만 허용
            if source_count.get(source, 0) >= 3:
                continue
            
            source_count[source] = source_count.get(source, 0) + 1
            result.append(doc)
        
        return result
    
    async def step2_generate_with_exaone(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """2단계: EXAONE으로 RAG 기반 답변 생성"""
        logger.info("2단계: EXAONE으로 답변 생성 중...")
        
        try:
            # 검색된 문서들을 컨텍스트로 구성
            context_parts = []
            total_score = 0
            
            for i, doc in enumerate(retrieved_docs):
                text = doc.get('text', '').strip()
                source = doc.get('source_file', 'unknown')
                score = doc.get('score', 0)
                keyword_score = doc.get('keyword_score', 0)
                
                total_score += score
                
                # 문서별 관련성 표시
                relevance_indicator = ""
                if score > 0.8:
                    relevance_indicator = "🔥 높은 관련성"
                elif score > 0.6:
                    relevance_indicator = "⭐ 중간 관련성"
                else:
                    relevance_indicator = "📄 참고 정보"
                
                # 구조화된 컨텍스트 구성
                context_part = f"""
                ==================== 참고문서 {i+1} ====================
                📋 출처: {source}
                {relevance_indicator} (유사도: {score:.3f}, 키워드 매칭: {keyword_score:.3f})

                📝 내용:
                {text}
                ================================================
                """
                context_parts.append(context_part)
            
            # 전체 컨텍스트 요약 정보 추가
            avg_score = total_score / len(retrieved_docs) if retrieved_docs else 0
            context_summary = f"""
            📊 컨텍스트 요약:
            - 총 참고문서 수: {len(retrieved_docs)}개
            - 평균 관련성 점수: {avg_score:.3f}
            - 신뢰도 수준: {"높음" if avg_score > 0.7 else "보통" if avg_score > 0.5 else "낮음"}

            """
            
            context = context_summary + "\n".join(context_parts)
            
            # 개선된 시스템 프롬프트 + 사용자 프롬프트
            system_prompt = """당신은 전문적이고 정확한 답변을 제공하는 AI 어시스턴트입니다. 

            다음 원칙에 따라 답변하세요:
            1. 제공된 참고 문서의 정보만을 사용하여 답변합니다
            2. 정확하고 구체적인 정보를 제공합니다  
            3. 불확실한 정보는 명시적으로 표현합니다
            4. 답변은 논리적이고 체계적으로 구성합니다
            5. 한국어로 자연스럽고 이해하기 쉽게 작성합니다
            6. 참고 문서에 없는 내용은 추측하지 않습니다
            """

            generation_prompt = f"""
            
            <|system|>
            {system_prompt}

            <|user|>
            다음 참고 문서를 바탕으로 사용자의 질문에 답변해주세요.

            **사용자 질문**: {query}

            **참고 문서**:
            {context}

            위 참고 문서의 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요. 답변은 다음 형식으로 구성해주세요:

            1. 핵심 답변 (간단명료하게)
            2. 상세 설명 (필요시)
            3. 추가 정보나 주의사항 (있는 경우)

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
                # 대안: 기본 답변 생성
                answer = f"""질문 '{query}'에 대한 답변을 다음 문서들을 바탕으로 제공합니다.

참고 문서: {len(retrieved_docs)}개 문서에서 검색

{context[:500]}{'...' if len(context) > 500 else ''}

※ EXAONE 모델 로딩 실패로 기본 답변을 제공합니다."""
            
            logger.info("EXAONE 답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.error(f"EXAONE 답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def step3_review_with_gpt4(self, query: str, exaone_answer: str, 
                                   retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """3단계: GPT-4로 최종 답변 검수"""
        logger.info("3단계: GPT-4로 답변 검수 중...")
        
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
                model=self.config.gpt_model_name,
                messages=[
                    {"role": "system", "content": "당신은 AI 답변을 검수하는 전문가입니다. 객관적이고 건설적인 피드백을 제공해주세요."},
                    {"role": "user", "content": review_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            try:
                review_result = json.loads(response.choices[0].message.content)
                logger.info(f"GPT-4 검수 완료: 전체 점수 {review_result.get('overall_score', 0)}/10")
                return review_result
            except json.JSONDecodeError:
                logger.warning("GPT-4 응답 JSON 파싱 실패")
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
            logger.error(f"GPT-4 검수 실패: {e}")
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
        
        # RAG 신뢰도 평가
        confidence_metrics = self._calculate_rag_confidence(query, retrieved_docs, exaone_answer)
        
        # 검수 결과에 따라 최종 답변 결정
        final_answer = exaone_answer
        gpt_score = gpt_review.get('overall_score', 5)
        rag_confidence = confidence_metrics.get('overall_confidence', 5.0)
        
        # 종합 신뢰도 점수 (GPT 검수 + RAG 신뢰도)
        combined_confidence = (gpt_score * 10 + rag_confidence) / 2
        
        # 신뢰도 기반 답변 보완
        confidence_level = confidence_metrics.get('confidence_level', '보통')
        
        if combined_confidence < 60:
            warning = f"""
⚠️ 신뢰도 주의: 이 답변의 신뢰도는 {combined_confidence:.1f}점({confidence_level})입니다.
- 검색 품질: {confidence_metrics.get('retrieval_quality', 0):.1f}/10
- 키워드 일치: {confidence_metrics.get('keyword_consistency', 0):.1f}/10
- 소스 다양성: {confidence_metrics.get('source_diversity', 0):.1f}/10
- 답변 완성도: {confidence_metrics.get('answer_completeness', 0):.1f}/10

추가 검증이 필요할 수 있습니다."""
            final_answer += warning
        elif combined_confidence >= 80:
            quality_note = f"""
✅ 높은 신뢰도: 이 답변의 신뢰도는 {combined_confidence:.1f}점({confidence_level})입니다."""
            final_answer += quality_note
        
        # 참고 문서 정보 추가
        source_files = list(set(doc.get('source_file', 'unknown') for doc in retrieved_docs))
        if source_files:
            final_answer += f"\n\n📚 참고 문서 ({len(source_files)}개): {', '.join(source_files[:3])}"
            if len(source_files) > 3:
                final_answer += f" 외 {len(source_files)-3}개"
        
        return {
            "query": query,
            "final_answer": final_answer,
            "confidence_score": combined_confidence,
            "confidence_level": confidence_level,
            "confidence_metrics": confidence_metrics,
            "approved": gpt_review.get('approved', False),
            "gpt_review_score": gpt_score,
            "rag_confidence_score": rag_confidence,
            "retrieved_documents": retrieved_docs,
            "sources": source_files,
            "pipeline_metadata": {
                "retrieved_count": len(retrieved_docs),
                "gpt4_review_score": gpt_score,
                "rag_confidence_breakdown": confidence_metrics,
                "total_processing_steps": 4
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
            
            # 2단계: EXAONE으로 답변 생성
            exaone_answer = await self.step2_generate_with_exaone(query, retrieved_docs)
            
            # 3단계: GPT-4로 검수
            gpt_review = await self.step3_review_with_gpt4(query, exaone_answer, retrieved_docs)
            
            # 4단계: 최종 답변 확정
            final_response = await self.step4_finalize_response(
                query, exaone_answer, gpt_review, retrieved_docs
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            final_response['processing_time'] = processing_time
            
            logger.info(f"복합 RAG 파이프라인 완료 (처리시간: {processing_time:.2f}초)")
            return final_response
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}")
            return {
                "query": query,
                "final_answer": f"파이프라인 실행 중 오류가 발생했습니다: {str(e)}",
                "confidence_score": 0,
                "confidence_level": '보통',
                "confidence_metrics": {},
                "approved": False,
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "composite_rag_initialized": True,
            "models_loaded": {
                "embedding": hasattr(self, 'embed_model'),
                "exaone": bool(self.exaone_model or self.exaone_pipeline),
                "openai_configured": bool(self.openai_client)
            },
            "config": {
                "embedding_model": self.config.embedding_model_name,
                "exaone_model": self.config.exaone_model_name,
                "gpt_model": self.config.gpt_model_name,
                "top_k_documents": self.config.top_k_documents
            }
        }

    def _calculate_rag_confidence(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                                 generated_answer: str) -> Dict[str, Any]:
        """RAG 신뢰도 종합 평가"""
        try:
            confidence_metrics = {}
            
            # 1. 문서 검색 품질 점수
            if retrieved_docs:
                scores = [doc.get('score', 0) for doc in retrieved_docs]
                avg_similarity = sum(scores) / len(scores)
                confidence_metrics['retrieval_quality'] = min(avg_similarity * 10, 10)  # 0-10 스케일
            else:
                confidence_metrics['retrieval_quality'] = 0
                
            # 2. 키워드 매칭 점수
            query_keywords = self._extract_keywords(query)
            answer_keywords = self._extract_keywords(generated_answer)
            keyword_overlap = self._calculate_keyword_overlap(query_keywords, answer_keywords)
            confidence_metrics['keyword_consistency'] = keyword_overlap * 10
            
            # 3. 문서 다양성 점수
            sources = set(doc.get('source_file', '') for doc in retrieved_docs)
            diversity_score = min(len(sources) * 2, 10)  # 최대 5개 소스까지 점수 부여
            confidence_metrics['source_diversity'] = diversity_score
            
            # 4. 답변 완성도 점수 (길이와 구조 기반)
            answer_length_score = min(len(generated_answer) / 200, 1) * 10  # 200자 기준
            structure_score = self._evaluate_answer_structure(generated_answer)
            confidence_metrics['answer_completeness'] = (answer_length_score + structure_score) / 2
            
            # 5. 전체 신뢰도 점수 계산
            weights = {
                'retrieval_quality': 0.3,
                'keyword_consistency': 0.2,
                'source_diversity': 0.2,
                'answer_completeness': 0.3
            }
            
            overall_confidence = sum(
                confidence_metrics[metric] * weight 
                for metric, weight in weights.items()
            )
            
            confidence_metrics['overall_confidence'] = round(overall_confidence, 2)
            confidence_metrics['confidence_level'] = self._get_confidence_level(overall_confidence)
            
            return confidence_metrics
            
        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {e}")
            return {
                'retrieval_quality': 5.0,
                'keyword_consistency': 5.0,
                'source_diversity': 5.0,
                'answer_completeness': 5.0,
                'overall_confidence': 5.0,
                'confidence_level': '보통'
            }
    
    def _evaluate_answer_structure(self, answer: str) -> float:
        """답변의 구조적 품질 평가"""
        structure_score = 0
        
        # 기본 점수
        if len(answer) > 50:
            structure_score += 3
            
        # 문장 구조 확인
        sentences = answer.split('.')
        if len(sentences) >= 2:
            structure_score += 2
            
        # 논리적 연결어 확인
        connectors = ['따라서', '그러므로', '또한', '하지만', '그런데', '예를 들어']
        if any(conn in answer for conn in connectors):
            structure_score += 2
            
        # 참고문서 언급 확인
        if '참고' in answer or '문서' in answer:
            structure_score += 1
            
        # 구체적 정보 포함 확인 (숫자, 날짜 등)
        import re
        if re.search(r'\d+', answer):
            structure_score += 2
            
        return min(structure_score, 10)
    
    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 점수를 레벨로 변환"""
        if score >= 8.0:
            return "매우 높음"
        elif score >= 6.5:
            return "높음"
        elif score >= 5.0:
            return "보통"
        elif score >= 3.0:
            return "낮음"
        else:
            return "매우 낮음"


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
    return await system.get_system_status() 