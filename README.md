# FlowMate

**기업급 로컬 AI 어시스턴트 - 직장 생산성 향상을 위한 솔루션**

안전한 온프레미스 배포를 위해 설계된 종합 AI 플랫폼으로, 데이터 프라이버시를 보장하면서 지능형 문서 분석, 프레젠테이션 피드백, HR 평가 기능을 제공합니다.

## 아키텍처 개요

FlowMate는 최신 로컬 AI 기술을 활용하여 귀하의 인프라 내에서 완전히 작동하는 엔터프라이즈급 솔루션을 제공합니다:

- **로컬 언어 모델**: 자연어 처리를 위한 Ollama Qwen2.5:7B
- **임베딩 엔진**: 의미 이해 및 벡터 표현을 위한 BGE-M3
- **벡터 데이터베이스**: 효율적인 유사도 검색 및 문서 검색을 위한 Qdrant
- **머신러닝**: HR 성과 예측을 위한 85% 정밀도의 RandomForest
- **미디어 처리**: MediaPipe 및 Librosa를 활용한 고급 비디오/오디오 분석

## 핵심 기능

### 지능형 문서 어시스턴트
- **다중 포맷 지원**: PDF, DOCX, TXT, CSV, XLSX, 이미지
- **맥락적 Q&A**: 대화 기억 기능을 갖춘 RAG 기반 검색
- **자동 생성**: 한국어 전문 보고서 및 프레젠테이션
- **실시간 처리**: 즉시 문서 벡터화 및 인덱싱

### 프레젠테이션 분석 엔진
- **컴퓨터 비전**: MediaPipe를 사용한 자세 및 제스처 분석
- **오디오 인텔리전스**: 음성 톤, 속도, 발음 평가
- **콘텐츠 분석**: 구조 및 흐름 평가
- **종합 피드백**: AI 생성 개선 권장사항

### HR 성과 예측
- **머신러닝 모델**: 85% 정확도의 RandomForest 분류기
- **다요소 분석**: 경험, 프로젝트, 급여를 포함한 10개 이상의 직원 지표
- **실시간 평가**: 즉시 성과 예측
- **데이터 기반 인사이트**: 인재 평가를 위한 과학적 접근법

## 기술적 기반

### 백엔드 인프라
```
Django 5.2.4          # 웹 프레임워크
LangChain 0.3.27       # LLM 오케스트레이션
LangGraph 0.6.5+       # 워크플로우 자동화
Qdrant Client 1.15.1   # 벡터 데이터베이스
```

### AI 및 머신러닝
```
PyTorch 2.5.1          # 딥러닝 프레임워크
Transformers 4.54.1    # Hugging Face 모델
Scikit-learn 1.7.1     # 클래식 ML 알고리즘
MediaPipe 0.10.21      # 컴퓨터 비전
Librosa 0.11.0         # 오디오 분석
```

### 데이터 처리
```
LangChain Community    # 문서 로더
PDFMiner.six          # PDF 추출
Python-docx           # Word 문서
OpenPyXL              # Excel 처리
Pillow                # 이미지 처리
```

## 빠른 시작

### 시스템 요구사항
- Python 3.10+
- CUDA 호환 GPU (권장)
- Ollama 런타임 환경
- 최적 성능을 위한 8GB 이상의 RAM

### 설치 방법

1. **저장소 복제**
```bash
git clone https://github.com/your-team/FlowMate.git
cd FlowMate
```

2. **환경 설정**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Ollama 모델 설치**
```bash
ollama pull qwen2.5:7b
```

4. **데이터베이스 마이그레이션**
```bash
python manage.py migrate
```

5. **애플리케이션 실행**
```bash
python manage.py runserver
```

FlowMate에 접속하려면 `http://localhost:8000`을 방문하세요.

## 시스템 아키텍처

### 워크플로우 엔진
FlowMate는 정교한 상태 기반 워크플로우 시스템을 구현합니다:

```
의도 분류 → 문서 검색 → 작업 처리 → 품질 검증 → 출력 생성
```

### 보안 설계
- **외부 종속성 없음**: 모든 처리가 로컬에서 수행
- **데이터 격리**: 클라우드 API 호출이나 외부 데이터 전송 없음
- **기업 규정 준수**: 기업 보안 요구사항 충족
- **감사 추적**: 모든 작업의 완전한 로깅

### 성능 최적화
- **효율적인 벡터화**: 최적화된 BGE-M3 임베딩 파이프라인
- **스마트 캐싱**: 지능형 문서 및 모델 캐싱
- **병렬 처리**: 빠른 응답을 위한 멀티스레드 작업
- **메모리 관리**: 장기 실행 기업 배포에 최적화

## 사용 사례

### 기업 교육
- **신입 사원 온보딩**: 상호작용형 문서 기반 학습
- **지식 관리**: 중앙화된 기업 지식 베이스
- **역량 평가**: AI 기반 평가 및 피드백

### 임원 보고
- **문서 통합**: 여러 소스로부터 자동 보고서 생성
- **임원 요약**: 핵심 통찰력 추출 및 형식화
- **프레젠테이션 준비**: 콘텐츠 구조화 및 전달 코칭

### HR 운영
- **성과 분석**: 데이터 기반 직원 평가
- **인재 파이프라인**: 경력 개발을 위한 예측 모델링
- **객관적 평가**: 편향 없는 성과 측정

## API 참조

### 문서 처리
```python
POST /upload/
Content-Type: multipart/form-data
Response: {"success": true, "file_path": "temp/document.pdf"}
```

### 채팅 인터페이스
```python
POST /ask/
Content-Type: application/json
Body: {"message": "query", "file_path": "temp/document.pdf"}
Response: {"answer": "AI response"}
```

### HR 예측
```python
POST /hr_evaluation/predict/
Content-Type: application/json
Body: {employee_metrics}
Response: {"result": "우수", "success": true}
```

### 프레젠테이션 분석
```python
POST /presentation/analyze/
Content-Type: application/json
Body: {"video_path": "path", "options": {}}
Response: {analysis_results}
```

## 개발팀

**FlowMate AI 연구개발팀**

| 팀원 | 역할 | 연락처 | GitHub |
|-------------|------|---------|--------|
| 고정현 | AI 엔지니어 | spellrain@naver.com | [@k-j-hyun](https://github.com/k-j-hyun) |
| 김도현 | AI 엔지니어 | kimdohyun222@naver.com | [@starfish99600](https://github.com/starfish99600) |
| 정종혁 | AI 엔지니어 | devna0111@gmail.com | [@devna0111](https://github.com/devna0111) |
| 장슬찬 | AI 엔지니어 | jsc980115@naver.com | [@jangseulchan](https://github.com/jangseulchan) |
| 박선우 | 데이터 분석 | du5968@daum.net | [@gulbiworker](https://github.com/gulbiworker) |

## 기업 기능

### 확장성
- **다중 사용자 지원**: 격리된 컨텍스트를 갖춘 동시 사용자 세션
- **로드 밸런싱**: 수평적 확장 기능
- **리소스 관리**: 구성 가능한 메모리 및 CPU 할당

### 모니터링 및 분석
- **성능 지표**: 실시간 시스템 모니터링
- **사용량 분석**: 상세한 사용자 상호작용 추적
- **오류 보고**: 포괄적인 로깅 및 알림

### 커스터마이징
- **모델 파인튜닝**: 도메인별 모델 적응
- **UI 테마**: 기업 브랜딩 및 스타일링
- **워크플로우 구성**: 맞춤형 비즈니스 프로세스 통합

## 기여하기

커뮤니티의 기여를 환영합니다. 기여 가이드라인을 읽고 검토를 위해 풀 리퀘스트를 제출해 주세요.

### 개발 가이드라인
- PEP 8 스타일 가이드라인 준수
- 새로운 기능에 대한 포괄적인 테스트 포함
- 모든 공개 API 문서화
- 하위 호환성 보장

## 라이선스

FlowMate는 기업용으로 개발된 독점 소프트웨어입니다. 라이선스 정보 및 상업적 배포 옵션에 대해서는 저희 팀에 문의하시기 바랍니다.

## 지원

기업 지원, 통합 지원 또는 맞춤 개발에 대해서는:

**이메일**: spellrain@naver.com  
**문서**: [프로젝트 위키](https://github.com/your-team/FlowMate/wiki)  
**이슈**: [GitHub 이슈](https://github.com/your-team/FlowMate/issues)

---

**FlowMate** - 로컬 AI 혁신을 통한 기업 우수성 강화