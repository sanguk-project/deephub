# 🚀 **ILJoo Deep Hub RAG System v2.1**

**LlamaIndex + FAISS + EXAONE 3.5 + GPT-4** 기반의 차세대 RAG(Retrieval-Augmented Generation) 시스템입니다.

## ✨ **주요 업그레이드 (v2.1)**

### 🎯 **로컬 모델 관리 시스템**
- **자동 모델 다운로드**: EXAONE 모델을 `/mnt/ssd/1/hub`에 자동 저장
- **로컬 우선 로딩**: 로컬 모델 존재 시 HuggingFace 대신 로컬에서 빠른 로딩
- **모델 무결성 검사**: 필수 파일(config.json, tokenizer.json, 모델 파일) 자동 확인
- **HuggingFace 폴백**: 로컬 모델 다운로드 실패 시 자동 폴백

### 🧠 **휘발성 메모리 기반 문서 저장소**
- **디스크 저장 제거**: 모든 문서를 메모리에서만 관리
- **개인정보 보호**: 서버 재시작 시 데이터 자동 초기화
- **실시간 관리**: 메모리 기반 업로드/삭제/목록 관리
- **통계 정보**: 저장된 문서 수, 메모리 사용량 실시간 확인

### 🎨 **마크다운 렌더링 웹 인터페이스**
- **실시간 마크다운**: marked.js 기반 답변 렌더링
- **코드 하이라이팅**: highlight.js를 활용한 GitHub 스타일 코드 블록
- **타이핑 효과**: 실시간 답변 표시와 마크다운 렌더링 결합
- **반응형 디자인**: 데스크톱/모바일 최적화 UI

### 🔄 **LangChain 최신화**
- **Deprecation 해결**: langchain-huggingface 마이그레이션
- **파이프라인 개선**: RunnableSequence(prompt | llm | output_parser) 방식
- **invoke() 메서드**: run() 대신 최신 invoke() 사용
- **StrOutputParser**: 체계적인 출력 파싱

### ⚡ **기존 v2.0 기능들**
- **3단계 검증 시스템**: RAG 검색 → EXAONE 3.5 생성 → GPT-4 검수
- **다중 기준 문서 검색**: 의미적 유사도(70%) + 키워드 매칭(30%)
- **동적 품질 관리**: 신뢰도 기반 적응형 피드백
- **소스 다양성 제어**: 최대 3개 문서/소스로 편향 방지

## 📚 **RAG(Retrieval-Augmented Generation)란?**

RAG는 검색 기반 생성 모델로, 기존 문서에서 관련 정보를 검색한 후 이를 바탕으로 정확한 답변을 생성하는 AI 기술입니다.

### 🔍 **복합 RAG 작동 원리**
1. **문서 업로드**: 메모리 기반 실시간 저장 및 LlamaIndex 인덱싱
2. **고급 문서 검색**: 의미적 + 키워드 매칭으로 관련 문서 발견
3. **품질 필터링**: 유사도, 텍스트 길이, 소스 다양성 기준 적용
4. **EXAONE 3.5 생성**: 로컬 한국어 특화 모델로 1차 답변 생성
5. **GPT-4 검수**: 품질, 정확성, 완성도 종합 검증
6. **마크다운 답변**: 타이핑 효과와 함께 렌더링된 최종 답변 제공

## 🏗️ **시스템 아키텍처**
![사내 AI Agent 설계도](https://github.com/user-attachments/assets/5454806f-d8ef-4e82-8cb8-c436f5dfa89d)

### 📁 **프로젝트 구조 (v2.1)**

```
ai_agent/deephub/
├── 🔧 admin/                           # 관리자 도구
│   └── tools/
│       └── document_indexer.py         # LlamaIndex 기반 문서 인덱싱
│
├── 👥 service/                         # 사용자 서비스
│   ├── api/
│   │   └── main.py                     # FastAPI 서버
│   ├── core/
│   │   ├── composite_rag_system.py     # EXAONE + GPT-4 복합 파이프라인
│   │   └── rag_system.py               # 기본 RAG + 로컬 모델 관리
│   ├── storage/
│   │   ├── vector_store.py             # FAISS IVF 벡터 검색
│   │   ├── memory_store.py             # 휘발성 메모리 문서 저장소
│   │   └── rag_logger.py               # RAG 성능 로깅
│   ├── indexing/
│   │   └── document_indexer.py         # 메모리 기반 문서 인덱싱
│   └── web/
│       ├── template/index.html         # 마크다운 렌더링 UI
│       └── static/                     # CSS/JS/라이브러리
│
├── 📊 shared/                          # 공용 설정 및 데이터
│   ├── config/
│   │   └── settings.py                 # 로컬 모델 경로 설정
│   ├── utils.py                        # 공용 유틸리티
│   └── data/
│       ├── documents/                  # 인덱싱할 문서들 (선택적)
│       └── faiss_index/                # IVF 벡터 인덱스
│
├── 🏠 sllm/web/                       # 메인 웹 인터페이스
│   ├── app.py                          # Flask 메인 서버
│   ├── template/index.html             # 사용자 친화적 UI
│   └── static/                         # 마크다운/하이라이트 리소스
│
├── ⚡ startup_optimization.py          # 시작 성능 최적화
├── 🚀 run.py                          # 메인 서비스 실행
└── 📚 README.md                       # 이 문서
```

### 🔄 **데이터 플로우 (v2.1)**

```mermaid
graph TD
    A[문서 업로드] --> B[메모리 저장소]
    B --> C[LlamaIndex 파싱]
    C --> D[BGE-M3 임베딩]
    D --> E[FAISS 벡터 저장]
    
    F[사용자 질문] --> G[임베딩 변환]
    G --> H[벡터 유사도 검색]
    H --> I[컨텍스트 추출]
    
    I --> J[로컬 EXAONE 모델]
    J --> K[1차 답변 생성]
    K --> L[GPT-4 검수]
    L --> M[마크다운 렌더링]
    M --> N[타이핑 효과 답변]
```

## 🚀 **빠른 시작**

### 1. 환경 설정

```bash
# 필수 환경변수 설정
export OPENAI_API_KEY="your-openai-api-key-here"

# EXAONE 모델 자동 다운로드 경로 (기본값)
# /mnt/ssd/1/hub/EXAONE-3.5-2.4B-Instruct
```

### 2. 서비스 시작 (3-5초 빠른 시작!)

```bash
cd /mnt/ssd/1/sanguk/ai_agent/deephub
python run.py

# 또는 웹 인터페이스만 사용
cd /mnt/ssd/1/sanguk/ai_agent/sllm/web
python app.py
```

### 3. 접속 및 사용

```bash
# 메인 웹 인터페이스 (휘발성 저장소 + 마크다운)
http://localhost:9999/

# DeepHub 고급 기능
http://localhost:8080/

# API 문서 (Swagger)
http://localhost:8080/docs

# 시스템 상태 확인
http://localhost:8080/status
```

## 🎯 **사용법**

### 1. 휘발성 문서 업로드

```bash
# 웹 인터페이스 사용 (권장)
# 1. http://localhost:9999 접속
# 2. "파일 선택" 버튼으로 문서 업로드
# 3. 자동 메모리 저장 및 인덱싱
# 4. 서버 재시작 시 모든 데이터 삭제됨

# 지원 형식: PDF, DOCX, TXT, MD
# 특징: 디스크 저장 없음, 개인정보 보호
```

### 2. 마크다운 질의응답

#### 웹 인터페이스 (권장)
- 브라우저에서 `http://localhost:9999` 접속
- 질문 입력 후 실시간 마크다운 답변 확인
- 코드 블록, 테이블, 리스트 자동 렌더링
- 타이핑 효과로 답변 점진적 표시

#### API 사용
```bash
# 로컬 EXAONE + GPT-4 복합 파이프라인
curl -X POST "http://localhost:8080/ask-composite" \
     -H "Content-Type: application/json" \
     -d '{"question": "AI의 발전 과정은?"}'

# 메모리 문서 관리
curl -X GET "http://localhost:9999/api/documents"      # 문서 목록
curl -X DELETE "http://localhost:9999/api/documents/1" # 문서 삭제
curl -X DELETE "http://localhost:9999/api/clear"       # 전체 삭제
```

## ⚙️ **설정 관리**

### 주요 설정 (`shared/config/settings.py`)

```python
@dataclass
class ModelConfig:
    """모델 설정"""
    embedding_model: str = "BAAI/bge-m3"
    verification_model: str = "gpt-4.1-2025-04-14"
    final_answer_model: str = "gpt-4.1-2025-04-14"
    exaone_model_path: str = "/mnt/ssd/1/hub/EXAONE-3.5-2.4B-Instruct"  # 로컬 저장 경로
    exaone_model_name: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"     # HuggingFace 모델명

@dataclass
class RAGConfig:
    """RAG 설정"""
    similarity_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_new_tokens: int = 1024      # EXAONE 모델 생성 길이
    temperature: float = 0.1        # 생성 온도
    llm_model: str = "exaone"       # LLM 모델 타입
```

### 로컬 모델 관리

```python
# 자동 모델 다운로드 및 저장
def _download_and_save_model(local_path, model_name):
    """EXAONE 모델을 HuggingFace에서 다운로드하여 로컬에 저장"""
    
def _check_model_exists(local_path):
    """로컬 모델 파일 무결성 확인"""
    
# 첫 실행 시:
# 1. 로컬 모델 존재 확인
# 2. 없으면 HuggingFace에서 다운로드
# 3. 로컬에 저장 후 다음부터 로컬 사용
```

### 휘발성 메모리 저장소

```python
class InMemoryDocumentStore:
    """메모리 기반 문서 저장소"""
    
    def store_document(self, doc_id, content, metadata):
        """문서를 메모리에 저장"""
        
    def get_document(self, doc_id):
        """문서 조회"""
        
    def remove_document(self, doc_id):
        """문서 삭제"""
        
    def clear_all(self):
        """모든 문서 삭제"""
        
    def get_stats(self):
        """저장소 통계 정보"""
```

## 📊 **성능 개선 사항**

### Before vs After (v2.1)

| 항목 | Before (v1.x) | After (v2.1) | 개선율 |
|------|---------------|---------------|--------|
| 시작 시간 | 30-60초 | 3-5초 | **90% 개선** |
| 모델 로딩 | 매번 HuggingFace | 로컬 캐시 우선 | **80% 빠름** |
| 문서 저장 | 디스크 기반 | 메모리 기반 | **100% 휘발성** |
| 답변 렌더링 | 일반 텍스트 | 마크다운 + 타이핑 | **UX 향상** |
| 벡터 검색 | Flat Cosine | IVF Algorithm | **30-50% 빠름** |
| 메모리 사용량 | 기본 | IVF 최적화 | **20-30% 절약** |
| 문서 처리 | LangChain | LlamaIndex 통합 | **단일화** |
<<<<<<< HEAD
| LLM 파이프라인 | 구버전 | 최신 LangChain | **Deprecation 해결** |
=======
| 라이브러리 수 | 다중 의존성 | 정리된 구조 | **유지보수성 향상** |
| 답변 품질 | GPT-4.1 단일 | EXAONE + GPT-4.1 | **다층 검증** |
>>>>>>> 72cf8176d80cdc2c3ce1360907eaf3a762c30674

### v2.1 신규 기능

- ✅ **로컬 모델 자동 관리**: EXAONE 모델 다운로드/저장/무결성 확인
- ✅ **휘발성 메모리 저장소**: 개인정보 보호를 위한 메모리 전용 문서 관리
- ✅ **마크다운 실시간 렌더링**: 코드 하이라이팅과 타이핑 효과
- ✅ **LangChain 최신화**: RunnableSequence 기반 파이프라인
- ✅ **개선된 웹 UI**: 반응형 디자인과 사용자 친화적 인터페이스

### v2.0 기존 기능

- ✅ **EXAONE 3.5 한국어 특화**: 로컬 모델로 생성 성능 향상
- ✅ **다중 기준 검색**: 의미적 + 키워드 매칭으로 정확도 개선
- ✅ **소스 다양성 제어**: 편향 방지 및 균형잡힌 답변
- ✅ **4가지 신뢰도 메트릭**: 종합적 품질 평가
- ✅ **LlamaIndex 통합**: 멀티 포맷 문서 처리
- ✅ **배치 인덱싱**: 효율적 대용량 처리

## 🔧 **API 엔드포인트**

### 휘발성 문서 관리 API (신규)

| 엔드포인트 | 방식 | 설명 | 특징 |
|------------|------|------|------|
| `/api/upload` | POST | **문서 업로드** | 메모리 저장 + 즉시 인덱싱 |
| `/api/documents` | GET | 저장된 문서 목록 | 메타데이터 포함 |
| `/api/documents/{id}` | DELETE | 개별 문서 삭제 | 실시간 삭제 |
| `/api/clear` | DELETE | 모든 문서 삭제 | 전체 초기화 |
| `/api/stats` | GET | 저장소 통계 | 문서 수, 메모리 사용량 |

### 핵심 RAG API

| 엔드포인트 | 방식 | 설명 | 특징 |
|------------|------|------|------|
| `/ask` | POST | **마크다운 질의응답** | EXAONE + 마크다운 렌더링 |
| `/ask-composite` | POST | 복합 RAG 파이프라인 | EXAONE → GPT-4 검수 |
| `/ask-langgraph` | POST | LangGraph 워크플로우 | 시각화 가능 |

### 시스템 관리 API

| 엔드포인트 | 방식 | 설명 |
|------------|------|------|
| `/status` | GET | 전체 시스템 상태 |
| `/status-composite` | GET | 복합 RAG 상태 |
| `/system/model-info` | GET | 로컬 모델 정보 |
| `/system/memory-stats` | GET | 메모리 저장소 상태 |

### 응답 형식

#### 마크다운 질의응답
```json
{
  "answer": "**마크다운 형식** 답변",
  "sources": ["document1.pdf", "document2.txt"],
  "confidence_score": 0.85,
  "processing_time": 2.3,
  "model_info": {
    "llm_model": "exaone",
    "source": "local",
    "path": "/mnt/ssd/1/hub/EXAONE-3.5-2.4B-Instruct"
  }
}
```

#### 복합 RAG 파이프라인
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

2. **EXAONE 모델 다운로드 오류**
   ```bash
   # 로그 확인
   tail -f server.log
   
   # 수동 다운로드 디렉토리 생성
   mkdir -p /mnt/ssd/1/hub
   chmod 755 /mnt/ssd/1/hub
   ```

3. **메모리 부족**
   ```bash
   # 모델 8bit 양자화 (settings.py)
   load_in_8bit = True
   
   # 더 적은 문서 검색
   similarity_top_k = 3
   ```

4. **문서 업로드 실패**
   - 지원 형식 확인: PDF, DOCX, TXT, MD
   - 파일 크기 제한: 50MB 이하 권장
   - 브라우저 새로고침 후 재시도

5. **마크다운 렌더링 오류**
   ```bash
   # CDN 연결 확인
   ping cdnjs.cloudflare.com
   
   # 로컬 리소스 확인
   ls ai_agent/sllm/web/static/
   ```

### 성능 최적화

```python
# 빠른 검색용 설정 (settings.py)
similarity_top_k = 3
max_new_tokens = 512
temperature = 0.1

# 정확도 우선 설정
similarity_top_k = 8
max_new_tokens = 1024
temperature = 0.05
```

### 로그 분석

```bash
# 시스템 로그 확인
tail -f ai_agent/deephub/server.log

# 자세한 디버깅
export LOG_LEVEL=DEBUG
python run.py

# 특정 컴포넌트 로그
grep "EXAONE" server.log
grep "memory_store" server.log
grep "markdown" server.log
```

## 🔮 **향후 계획 (v2.2)**

### 계획된 기능
- [ ] **GPU 가속**: FAISS GPU 버전 지원으로 검색 속도 향상
- [ ] **실시간 협업**: 다중 사용자 동시 문서 편집
- [ ] **음성 인터페이스**: STT/TTS 기반 음성 질의응답
- [ ] **다국어 모델**: 영어, 중국어, 일본어 지원 확장
- [ ] **모바일 앱**: React Native 기반 모바일 클라이언트

### 기술적 개선
- [ ] **스트리밍 응답**: 실시간 토큰 스트리밍으로 응답 체감 속도 향상
- [ ] **모델 양자화**: INT4/INT8 양자화로 메모리 사용량 50% 절약
- [ ] **캐싱 시스템**: Redis 기반 질의 결과 캐싱
- [ ] **분산 처리**: 다중 GPU/서버 분산 추론
- [ ] **자동 모델 업데이트**: 최신 EXAONE 버전 자동 감지/업데이트

---

## 📄 **라이선스**

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 👥 **기여하기**

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경 사항 커밋 (`git commit -m 'Add amazing feature'`)
4. Push to 브랜치 (`git push origin feature/amazing-feature`)
5. Pull Request 제출

### 기여 가이드라인
- 코드 스타일: PEP 8 준수
- 테스트: 새 기능에 대한 단위 테스트 작성
- 문서: README 및 API 문서 업데이트
- 이슈: 버그 리포트 시 재현 가능한 예제 포함

---

<<<<<<< HEAD
**🎯 ILJoo Deep Hub v2.1 - 로컬 모델 + 휘발성 저장소 + 마크다운 렌더링으로 더욱 안전하고 빠른 AI 질의응답을 경험하세요!** 
=======
**🎯 ILJoo Deep Hub v2.0 - 차세대 RAG 시스템으로 더 정확하고 빠른 AI 질의응답을 경험하세요!** 
>>>>>>> 72cf8176d80cdc2c3ce1360907eaf3a762c30674
