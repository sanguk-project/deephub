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
        # FAISS CPU 전용 설정 - GPU 경고 메시지 완전 억제
        os.environ['FAISS_NO_AVX2'] = '0'  # AVX2 지원 활성화
        os.environ['FAISS_OMP_NUM_THREADS'] = '8'  # OpenMP 스레드 수
        os.environ['FAISS_OPT_LEVEL'] = 'generic'  # GPU 기능 시도 방지
        os.environ['FAISS_DISABLE_CPU_FEATURES'] = ''  # CPU 기능은 모두 활성화
        
        # GPU 관련 설정 완전 비활성화
        if 'FAISS_ENABLE_GPU' in os.environ:
            del os.environ['FAISS_ENABLE_GPU']
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # PyTorch CPU 최적화
        os.environ['TORCH_NUM_THREADS'] = '8'
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
        
        # 임시 파일 최적화
        os.environ['TMPDIR'] = '/tmp'
        
        # HuggingFace 캐시 최적화
        os.environ['HF_HOME'] = '/tmp/huggingface_cache'
        
        logger.info("시작 최적화 환경변수 설정 완료 (FAISS CPU 전용, GPU 경고 억제)")
    
    @staticmethod
    def suppress_faiss_gpu_warnings():
        """FAISS GPU 경고 메시지 완전 억제"""
        try:
            # FAISS 관련 모든 로거를 CRITICAL로 설정
            faiss_loggers = [
                'faiss', 'faiss._loader', 'faiss.loader', 
                'faiss.python_callbacks', 'faiss.contrib'
            ]
            
            for logger_name in faiss_loggers:
                faiss_logger = logging.getLogger(logger_name)
                faiss_logger.setLevel(logging.CRITICAL)
                faiss_logger.propagate = False  # 상위 로거로 전파 방지
            
            logger.info("FAISS GPU 경고 메시지 억제 설정 완료")
            return True
        except Exception as e:
            logger.warning(f"FAISS 로깅 억제 설정 실패: {e}")
            return False
    
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
    
    # FAISS GPU 경고 메시지 억제
    optimizer.suppress_faiss_gpu_warnings()
    
    # 핵심 모듈 미리 로드
    optimizer.preload_critical_modules()
    
    return optimizer 