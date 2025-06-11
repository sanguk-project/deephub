"""
RAG 품질 스코어링 및 MongoDB 로깅 시스템
GPT-4.1을 사용하여 RAG 품질을 평가하고 결과를 MongoDB에 저장
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from shared.utils import get_openai_client
from shared.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class RAGEvaluationConfig:
    """RAG 평가 설정"""
    model_name: str = "gpt-4o"  # GPT-4.1 latest
    max_tokens: int = 500
    temperature: float = 0.1
    mongodb_enabled: bool = True

class RAGEvaluationResult(BaseModel):
    """RAG 평가 결과 모델"""
    user_prompt: str
    rag_context: str
    rag_answer: str
    relevance_score: int  # 0-10
    accuracy_score: int   # 0-10
    completeness_score: int  # 0-10
    overall_score: float  # 평균 점수
    evaluation_reason: str
    timestamp: datetime
    session_id: str

class RAGQualityEvaluator:
    """RAG 품질 평가기 (GPT-4.1 기반)"""
    
    def __init__(self, config: RAGEvaluationConfig = None):
        self.config = config or RAGEvaluationConfig()
        
        # OpenAI 클라이언트 초기화
        try:
            self.openai_client = get_openai_client(async_mode=True)
            logger.info("RAG 품질 평가기 초기화 완료 (GPT-4.1)")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
    
    async def evaluate_rag_quality(self, user_prompt: str, rag_context: str, rag_answer: str) -> RAGEvaluationResult:
        """RAG 품질을 GPT-4.1로 평가"""
        
        if not self.openai_client:
            logger.warning("OpenAI 클라이언트가 없어 기본 점수로 평가")
            return RAGEvaluationResult(
                user_prompt=user_prompt,
                rag_context=rag_context,
                rag_answer=rag_answer,
                relevance_score=5,
                accuracy_score=5,
                completeness_score=5,
                overall_score=5.0,  # 이미 소수 첫째자리
                evaluation_reason="OpenAI 클라이언트 없음 - 기본 점수",
                timestamp=datetime.now(),
                session_id="unknown"
            )
        
        # GPT-4.1을 이용한 평가 프롬프트
        evaluation_prompt = f"""
당신은 RAG(Retrieval-Augmented Generation) 시스템의 품질을 평가하는 전문가입니다.

다음 기준으로 RAG 시스템의 응답을 평가해주세요:

**사용자 질문**: {user_prompt}

**검색된 컨텍스트**: 
{rag_context}

**RAG 시스템의 답변**: 
{rag_answer}

**평가 기준** (각각 0-10점):
1. **관련성 (Relevance)**: 답변이 사용자 질문과 얼마나 관련이 있는가?
2. **정확성 (Accuracy)**: 검색된 컨텍스트를 바탕으로 답변이 얼마나 정확한가?
3. **완성도 (Completeness)**: 답변이 질문에 대해 충분히 완전한 정보를 제공하는가?

**응답 형식** (JSON):
{{
    "relevance_score": 8,
    "accuracy_score": 9,
    "completeness_score": 7,
    "evaluation_reason": "답변이 질문과 매우 관련이 있고 정확하지만, 일부 세부사항이 누락됨"
}}

평가를 시작하세요:
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "당신은 RAG 시스템 품질 평가 전문가입니다. 항상 JSON 형식으로 응답하세요."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            import json
            try:
                # JSON 블록 추출
                if "```json" in evaluation_text:
                    json_start = evaluation_text.find("```json") + 7
                    json_end = evaluation_text.find("```", json_start)
                    json_str = evaluation_text[json_start:json_end].strip()
                elif "{" in evaluation_text and "}" in evaluation_text:
                    json_start = evaluation_text.find("{")
                    json_end = evaluation_text.rfind("}") + 1
                    json_str = evaluation_text[json_start:json_end]
                else:
                    raise ValueError("JSON 형식을 찾을 수 없음")
                
                eval_data = json.loads(json_str)
                
                relevance_score = int(eval_data.get("relevance_score", 5))
                accuracy_score = int(eval_data.get("accuracy_score", 5))
                completeness_score = int(eval_data.get("completeness_score", 5))
                evaluation_reason = eval_data.get("evaluation_reason", "평가 완료")
                
                # 점수 범위 검증
                relevance_score = max(0, min(10, relevance_score))
                accuracy_score = max(0, min(10, accuracy_score))
                completeness_score = max(0, min(10, completeness_score))
                
                overall_score = (relevance_score + accuracy_score + completeness_score) / 3.0
                
            except Exception as e:
                logger.warning(f"GPT-4.1 평가 결과 파싱 실패: {e}")
                # 기본값 사용
                relevance_score = accuracy_score = completeness_score = 7
                overall_score = 7.0
                evaluation_reason = f"파싱 실패로 기본 점수 적용: {str(e)[:100]}"
            
            return RAGEvaluationResult(
                user_prompt=user_prompt,
                rag_context=rag_context,
                rag_answer=rag_answer,
                relevance_score=relevance_score,
                accuracy_score=accuracy_score,
                completeness_score=completeness_score,
                overall_score=round(overall_score, 1),  # 소수 첫째자리까지만
                evaluation_reason=evaluation_reason,
                timestamp=datetime.now(),
                session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        except Exception as e:
            logger.error(f"GPT-4.1 평가 중 오류: {e}")
            return RAGEvaluationResult(
                user_prompt=user_prompt,
                rag_context=rag_context,
                rag_answer=rag_answer,
                relevance_score=5,
                accuracy_score=5,
                completeness_score=5,
                overall_score=5.0,  # 이미 소수 첫째자리
                evaluation_reason=f"평가 오류: {str(e)[:100]}",
                timestamp=datetime.now(),
                session_id="error_session"
            )

class MongoDBLogger:
    """MongoDB RAG 로깅 시스템"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._setup_mongodb()
    
    def _setup_mongodb(self):
        """MongoDB 연결 설정"""
        try:
            # MongoDB 연결 시도 (pymongo 필요)
            try:
                import pymongo
                
                # 연결 문자열 (환경변수에서 가져오거나 기본값)
                mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
                
                self.client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
                # 연결 테스트
                self.client.server_info()
                
                self.db = self.client["rag_system"]
                self.collection = self.db["rag_evaluations"]
                
                logger.info("MongoDB 연결 성공")
                
            except ImportError:
                logger.warning("pymongo가 설치되지 않음. MongoDB 로깅 비활성화")
            except Exception as e:
                logger.warning(f"MongoDB 연결 실패: {e}. 로깅 비활성화")
                
        except Exception as e:
            logger.error(f"MongoDB 설정 오류: {e}")
    
    async def log_rag_evaluation(self, evaluation_result: RAGEvaluationResult) -> bool:
        """RAG 평가 결과를 MongoDB에 저장"""
        
        if self.collection is None:
            logger.debug("MongoDB가 설정되지 않아 로깅을 건너뜁니다")
            return False
        
        try:
            # 비동기 MongoDB 작업을 위해 별도 스레드에서 실행
            def _insert_document():
                document = {
                    "user_prompt": evaluation_result.user_prompt,
                    "rag_context": evaluation_result.rag_context,
                    "rag_answer": evaluation_result.rag_answer,
                    "relevance_score": evaluation_result.relevance_score,
                    "accuracy_score": evaluation_result.accuracy_score,
                    "completeness_score": evaluation_result.completeness_score,
                    "overall_score": round(evaluation_result.overall_score, 1),  # 소수 첫째자리까지만
                    "evaluation_reason": evaluation_result.evaluation_reason,
                    "timestamp": evaluation_result.timestamp,
                    "session_id": evaluation_result.session_id,
                    "created_at": datetime.now()
                }
                
                result = self.collection.insert_one(document)
                return result.inserted_id
            
            # 별도 스레드에서 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_insert_document)
                inserted_id = future.result(timeout=5)
            
            logger.info(f"RAG 평가 결과 MongoDB 저장 완료: {inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB 저장 중 오류: {e}")
            return False
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 평가 결과 조회"""
        if self.collection is None:
            return []
        
        try:
            results = list(self.collection.find().sort("timestamp", -1).limit(limit))
            # ObjectId를 문자열로 변환
            for result in results:
                result["_id"] = str(result["_id"])
            return results
        except Exception as e:
            logger.error(f"MongoDB 조회 중 오류: {e}")
            return []
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """평가 통계 조회"""
        if self.collection is None:
            return {}
        
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_evaluations": {"$sum": 1},
                        "avg_overall_score": {"$avg": "$overall_score"},
                        "avg_relevance_score": {"$avg": "$relevance_score"},
                        "avg_accuracy_score": {"$avg": "$accuracy_score"},
                        "avg_completeness_score": {"$avg": "$completeness_score"}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            if result:
                stats = result[0]
                del stats["_id"]
                return stats
            else:
                return {}
                
        except Exception as e:
            logger.error(f"통계 조회 중 오류: {e}")
            return {}

# 전역 인스턴스
_rag_evaluator = None
_mongodb_logger = None

def get_rag_evaluator() -> RAGQualityEvaluator:
    """RAG 품질 평가기 인스턴스 반환 (지연 로딩)"""
    global _rag_evaluator
    
    if _rag_evaluator is None:
        logger.info("RAG 품질 평가기 초기화 중...")
        _rag_evaluator = RAGQualityEvaluator()
        logger.info("RAG 품질 평가기 초기화 완료")
    
    return _rag_evaluator

def get_mongodb_logger() -> MongoDBLogger:
    """MongoDB 로거 인스턴스 반환 (지연 로딩)"""
    global _mongodb_logger
    
    if _mongodb_logger is None:
        logger.info("MongoDB 로거 초기화 중...")
        _mongodb_logger = MongoDBLogger()
        logger.info("MongoDB 로거 초기화 완료")
    
    return _mongodb_logger

# 편의 함수들
async def evaluate_and_log_rag(user_prompt: str, rag_context: str, rag_answer: str) -> RAGEvaluationResult:
    """RAG 품질 평가 및 MongoDB 로깅 (일괄 처리)"""
    
    # 평가 수행
    evaluator = get_rag_evaluator()
    evaluation_result = await evaluator.evaluate_rag_quality(user_prompt, rag_context, rag_answer)
    
    # MongoDB 로깅
    mongodb_logger = get_mongodb_logger()
    await mongodb_logger.log_rag_evaluation(evaluation_result)
    
    logger.info(f"RAG 평가 완료 - 전체 점수: {evaluation_result.overall_score}/10")
    
    return evaluation_result

async def quick_rag_score(user_prompt: str, rag_context: str, rag_answer: str) -> float:
    """RAG 품질 점수만 빠르게 반환"""
    evaluator = get_rag_evaluator()
    evaluation_result = await evaluator.evaluate_rag_quality(user_prompt, rag_context, rag_answer)
    return evaluation_result.overall_score 