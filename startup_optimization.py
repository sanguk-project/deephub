"""
시작 성능 최적화 유틸리티
"""

import os
import logging

logger = logging.getLogger(__name__)

class StartupOptimizer:
    """시작 성능 최적화 관리자"""
    
    @staticmethod
    def set_environment_optimizations():
        """환경 변수 최적화 설정"""
        # FAISS CPU 전용 설정
        os.environ['FAISS_NO_AVX2'] = '0'  # AVX2 지원 활성화
        os.environ['FAISS_OMP_NUM_THREADS'] = '8'  # OpenMP 스레드 수
        
        # GPU 관련 설정 비활성화
        if 'FAISS_ENABLE_GPU' in os.environ:
            del os.environ['FAISS_ENABLE_GPU']
        if 'FAISS_DISABLE_CPU_FEATURES' in os.environ:
            del os.environ['FAISS_DISABLE_CPU_FEATURES']
        
        # PyTorch CPU 최적화
        os.environ['TORCH_NUM_THREADS'] = '8'
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
        
        # 임시 파일 최적화
        os.environ['TMPDIR'] = '/tmp'
        
        # HuggingFace 캐시 최적화
        os.environ['HF_HOME'] = '/tmp/huggingface_cache'
        
        logger.info("시작 최적화 환경변수 설정 완료 (FAISS CPU 전용)")
    
    @staticmethod
    def preload_critical_modules():
        """중요한 모듈들만 미리 로드"""
        try:
            # 기본 필수 모듈들만 로드
            import json
            import uuid
            from pathlib import Path
            
            logger.info("핵심 모듈 미리 로드 완료")
            return True
        except Exception as e:
            logger.warning(f"모듈 미리 로드 실패: {e}")
            return False

def optimize_startup():
    """시작 최적화 실행"""
    optimizer = StartupOptimizer()
    
    # 환경 최적화
    optimizer.set_environment_optimizations()
    
    # 핵심 모듈 미리 로드
    optimizer.preload_critical_modules()
    
    return optimizer 