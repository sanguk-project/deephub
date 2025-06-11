# 🚀 **ILJoo Deep Hub RAG System v2.0**

**LlamaIndex + FAISS + EXAONE 3.5 + GPT-4** 기반의 차세대 RAG(Retrieval-Augmented Generation) 시스템입니다.

## ✨ **주요 업그레이드 (v2.0)**

### 🎯 **복합 RAG 파이프라인**
- **3단계 검증 시스템**: RAG 검색 → EXAONE 3.5 생성 → GPT-4 검수
- **다중 기준 문서 검색**: 의미적 유사도(70%) + 키워드 매칭(30%)
- **동적 품질 관리**: 신뢰도 기반 적응형 피드백
- **소스 다양성 제어**: 최대 3개 문서/소스로 편향 방지

### ⚡ **성능 최적화**
- **IVF 벡터 검색**: 기존 대비 30-50% 속도 향상, 20-30% 메모리 절약
- **지연 로딩**: 첫 요청 시에만 모델 로딩으로 3-5초 빠른 시작
- **LlamaIndex 통합**: 단일 라이브러리로 문서 처리 통일
- **캐시 최적화**: 불필요한 파일 및 import 정리

### 🧠 **EXAONE 3.5 통합**
- **구조화된 프롬프트**: `<|system|>`, `<|user|>`, `<|assistant|>` 태그 기반
- **6가지 생성 원칙**: 정확성, 완전성, 간결성, 구조화, 증거 기반, 사용자 중심
- **반복 억제**: repetition_penalty=1.1, length_penalty=1.0 적용
- **품질 지표**: 검색 품질(30%) + 키워드 일관성(20%) + 소스 다양성(20%) + 답변 완전성(30%)

### 📁 **스마트 문서 관리**
- **멀티 포맷 지원**: PDF, DOCX, TXT, MD, HTML, RTF
- **중복 방지**: 파일 해시 기반 자동 중복 제거
- **배치 처리**: 디렉토리 단위 효율적 인덱싱
- **LlamaIndex 리더**: 통일된 문서 추출 인터페이스

## 📚 **RAG(Retrieval-Augmented Generation)란?**

RAG는 검색 기반 생성 모델로, 기존 문서에서 관련 정보를 검색한 후 이를 바탕으로 정확한 답변을 생성하는 AI 기술입니다.

### 🔍 **복합 RAG 작동 원리**
1. **고급 문서 검색**: 의미적 + 키워드 매칭으로 관련 문서 발견
2. **품질 필터링**: 유사도, 텍스트 길이, 소스 다양성 기준 적용
3. **EXAONE 3.5 생성**: 한국어 특화 모델로 1차 답변 생성
4. **GPT-4 검수**: 품질, 정확성, 완성도 종합 검증
5. **최종 답변**: 신뢰도 점수와 함께 검증된 답변 제공

## 🏗️ **시스템 아키텍처**

## 📁 **프로젝트 구조 (v2.0)**

```
ai_agent/deephub/
├── 🔧 admin/                           # 관리자 도구
│   └── tools/
│       └── document_indexer.py         # LlamaIndex 기반 문서 인덱싱
│
├── 👥 service/                         # 사용자 서비스
│   ├── api/
│   │   └── main.py                     # FastAPI 서버 (불필요한 import 정리됨)
│   ├── core/                           # 핵심 RAG 시스템
│   │   ├── composite_rag_system.py     # EXAONE + GPT-4 복합 파이프라인
│   │   ├── rag_system.py               # 기본 RAG + GPT-4 검증
│   │   └── langgraph_rag.py            # LangGraph 워크플로우
│   ├── storage/                        # 데이터 저장소
│   │   ├── vector_store.py             # FAISS IVF 벡터 검색
│   │   └── rag_logger.py               # RAG 성능 로깅
│   └── web/                            # 웹 인터페이스
│       ├── template/index.html         # 메인 UI
│       └── static/                     # 정적 파일
│
├── 📊 shared/                          # 공용 설정 및 데이터
│   ├── config/
│   │   └── settings.py                 # 통합 설정 관리
│   ├── utils.py                        # 공용 유틸리티
│   └── data/
│       ├── documents/                  # 인덱싱할 문서들
│       └── faiss_index/                # IVF 벡터 인덱스
│
├── ⚡ startup_optimization.py          # 시작 성능 최적화
├── 🚀 run.py                          # 메인 서비스 실행
└── 📚 README.md                       # 이 문서
```

## 🚀 **빠른 시작**

### 1. 환경 설정

```bash
# 필수 환경변수 설정
export OPENAI_API_KEY="your-openai-api-key-here"

# EXAONE 모델 경로 확인 (필요시 settings.py에서 수정)
# 기본값: /mnt/ssd/1/sanguk/EXAONE-3.5-2.4B-Instruct
```

### 2. 서비스 시작 (3-5초 빠른 시작!)

```bash
cd /mnt/ssd/1/sanguk/ai_agent/deephub
python run.py
```

### 3. 접속 및 사용

```bash
# 메인 웹 인터페이스
http://localhost:9999/

# API 문서 (Swagger)
http://localhost:9999/docs

# 시스템 상태 확인
http://localhost:9999/status
```

## 🎯 **사용법**

### 1. 문서 추가 및 인덱싱

```bash
# 1단계: documents 폴더에 파일 추가
cp your_document.pdf shared/data/documents/

# 2단계: 서버 재시작 (자동 인덱싱)
python run.py

# 지원 형식: PDF, DOCX, TXT, MD, HTML, RTF
# 자동 중복 제거 및 배치 처리
```

### 2. 복합 RAG 질의응답

#### 웹 인터페이스 (권장)
- 브라우저에서 `http://localhost:9999` 접속
- 질문 입력 후 답변 확인
- 신뢰도 점수 및 소스 정보 제공

#### API 사용
```bash
# 복합 RAG 파이프라인 (EXAONE + GPT-4)
curl -X POST "http://localhost:9999/ask-composite" \
     -H "Content-Type: application/json" \
     -d '{"question": "AI의 발전 과정은?"}'

# 기본 RAG (GPT-4 검증)
curl -X POST "http://localhost:9999/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "머신러닝이란?"}'
```

## ⚙️ **설정 관리**

### 주요 설정 (`shared/config/settings.py`)

```python
# 모델 설정
embedding_model = "BAAI/bge-m3"                    # 임베딩 모델
verification_model = "gpt-4.1-2025-04-14"         # GPT-4 검증
final_answer_model = "gpt-4.1-2025-04-14"         # GPT-4 최종 답변
exaone_model_path = "/path/to/EXAONE-3.5-2.4B"    # EXAONE 모델

# RAG 성능 설정
max_retrieved_docs = 8                              # 검색 문서 수
similarity_threshold = 0.3                         # 유사도 임계값
keyword_weight = 0.3                               # 키워드 가중치
semantic_weight = 0.7                              # 의미적 가중치
diversity_threshold = 3                            # 소스별 최대 문서수

# 벡터 DB 설정 (IVF)
index_type = "IndexIVFFlat"                        # IVF 알고리즘
nlist = 256                                        # 클러스터 수
nprobe = 32                                        # 검색 클러스터 수
```

## 📊 **성능 개선 사항**

### Before vs After (v2.0)

| 항목 | Before (v1.x) | After (v2.0) | 개선율 |
|------|---------------|---------------|--------|
| 시작 시간 | 30-60초 | 3-5초 | **90% 개선** |
| 벡터 검색 | Flat Cosine | IVF Algorithm | **30-50% 빠름** |
| 메모리 사용량 | 기본 | IVF 최적화 | **20-30% 절약** |
| 문서 처리 | LangChain | LlamaIndex 통합 | **단일화** |
| 라이브러리 수 | 다중 의존성 | 정리된 구조 | **유지보수성 향상** |
| 답변 품질 | GPT-4 단일 | EXAONE + GPT-4 | **다층 검증** |

### 새로운 기능

- ✅ **EXAONE 3.5 한국어 특화**: 로컬 모델로 생성 성능 향상
- ✅ **다중 기준 검색**: 의미적 + 키워드 매칭으로 정확도 개선
- ✅ **소스 다양성 제어**: 편향 방지 및 균형잡힌 답변
- ✅ **4가지 신뢰도 메트릭**: 종합적 품질 평가
- ✅ **LlamaIndex 통합**: 멀티 포맷 문서 처리
- ✅ **배치 인덱싱**: 효율적 대용량 처리

## 🔧 **API 엔드포인트**

### 핵심 RAG API

| 엔드포인트 | 방식 | 설명 | 특징 |
|------------|------|------|------|
| `/ask-composite` | POST | **복합 RAG 파이프라인** | EXAONE → GPT-4 검수 |
| `/ask` | POST | 기본 RAG + GPT-4 검증 | 단일 검증 시스템 |
| `/ask-langgraph` | POST | LangGraph 워크플로우 | 시각화 가능 |

### 관리 API

| 엔드포인트 | 방식 | 설명 |
|------------|------|------|
| `/admin/index-text` | POST | 텍스트 직접 인덱싱 |
| `/admin/index-file` | POST | 파일 경로 인덱싱 |
| `/admin/upload-document` | POST | 파일 업로드 + 즉시 인덱싱 |
| `/status` | GET | 전체 시스템 상태 |
| `/status-composite` | GET | 복합 RAG 상태 |

### 응답 형식

```json
{
  "final_answer": "검증된 최종 답변",
  "confidence_score": 0.85,
  "sources": ["document1.pdf", "document2.txt"],
  "approved": true,
  "review_summary": "GPT-4 검수 요약",
  "processing_time": 2.3,
  "pipeline_metadata": {
    "retrieval_quality": 0.9,
    "keyword_consistency": 0.8,
    "source_diversity": 0.7,
    "answer_completeness": 0.85
  }
}
```

## 🛠️ **트러블슈팅**

### 일반적인 문제

1. **OpenAI API 키 오류**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   source ~/.bashrc
   ```

2. **EXAONE 모델 경로 오류**
   - `shared/config/settings.py`에서 `exaone_model_path` 수정

3. **메모리 부족**
   ```bash
   # IVF 설정 조정
   nlist = 128  # 클러스터 수 감소
   nprobe = 16  # 검색 범위 감소
   ```

4. **문서 인덱싱 실패**
   - 지원 형식 확인: PDF, DOCX, TXT, MD, HTML, RTF
   - 파일 권한 및 경로 확인

### 성능 최적화

```python
# 빠른 검색용 설정
max_retrieved_docs = 5
similarity_threshold = 0.4

# 정확도 우선 설정
max_retrieved_docs = 10
similarity_threshold = 0.2
keyword_weight = 0.4
```

## 🔮 **향후 계획**

- [ ] **GPU 가속**: FAISS GPU 버전 지원
- [ ] **실시간 인덱싱**: 파일 변경 감지 자동 업데이트
- [ ] **다국어 지원**: 다양한 언어 모델 통합
- [ ] **캐싱 시스템**: 질의 결과 캐싱으로 응답 속도 향상
- [ ] **클러스터링**: 대용량 문서 컬렉션 분산 처리

---

## 📄 **라이선스**

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 👥 **기여하기**

1. Fork 프로젝트
2. Feature 브랜치 생성
3. 변경 사항 커밋
4. Pull Request 제출

---

**🎯 ILJoo Deep Hub v2.0 - 차세대 RAG 시스템으로 더 정확하고 빠른 AI 질의응답을 경험하세요!** 