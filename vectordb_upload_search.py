import os
import hashlib
import re
from parsing_utils import split_chunks

# Qdrant import 시도
try:
    from langchain_qdrant import Qdrant
    QDRANT_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.vectorstores import Qdrant
        QDRANT_AVAILABLE = True
    except ImportError:
        QDRANT_AVAILABLE = False

from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from collections import deque

# 전역 캐시
_vector_store_cache = {}
_client_cache = None

class BufferMemory:
    """대화 히스토리 관리"""
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
    
    def append(self, user, assistant):
        self.history.append({"user": user, "assistant": assistant})
    
    def get_formatted_history(self):
        if not self.history:
            return ""
        return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history])

def get_file_hash(file_path: str) -> str:
    """파일 해시 생성 (크기+수정시간 기반)"""
    try:
        stat = os.stat(file_path)
        quick_hash = f"{stat.st_size}_{int(stat.st_mtime)}"
        return hashlib.md5(quick_hash.encode()).hexdigest()
    except:
        return hashlib.md5(file_path.encode()).hexdigest()

def get_qdrant_client():
    """Qdrant 클라이언트 캐싱"""
    global _client_cache
    if _client_cache is None:
        try:
            _client_cache = QdrantClient(host="localhost", port=6333)
        except Exception as e:
            print(f"[Qdrant 연결 실패: {e}]")
            return None
    return _client_cache

def get_llm(tokens=256):
    """LLM 인스턴스 생성 (호환성 처리)"""
    model_name = "anpigon/qwen2.5-7b-instruct-kowiki:latest"
    # model_name = "exaone3.5:latest"
    try:
        return ChatOllama(model=model_name, temperature=0.2, num_predict=tokens)
    except TypeError:
        try:
            return ChatOllama(model=model_name, temperature=0.2, max_tokens=tokens)
        except TypeError:
            return ChatOllama(model=model_name, temperature=0.2)

def data_to_vectorstore(file_path: str):
    """벡터스토어 - 캐싱 및 빠른 체크"""
    
    # 캐시 확인 (가장 빠른 경로)
    file_hash = get_file_hash(file_path)
    cache_key = f"{file_path}_{file_hash}"
    
    if cache_key in _vector_store_cache:
        print(f"[캐시에서 벡터스토어 로드: {file_path}]")
        return _vector_store_cache[cache_key]
    
    if not QDRANT_AVAILABLE:
        print("[Qdrant 사용 불가 - None 반환]")
        return None
    
    client = get_qdrant_client()
    if client is None:
        return None
    
    collection_name = f"doc_{file_hash}"
    
    # 기존 컬렉션 빠른 확인
    try:
        existing_collections = [col.name for col in client.get_collections().collections]
        
        if collection_name in existing_collections:
            print(f"[기존 컬렉션 사용: {collection_name}]")
            
            # 벡터 수 빠른 체크
            try:
                collection_info = client.get_collection(collection_name)
                print(collection_info)
                if collection_info.points_count  > 0:
                    vector_store = Qdrant(
                        client=client,
                        collection_name=collection_name,
                        embeddings=OllamaEmbeddings(model="bge-m3:567m")
                    )
                    
                    # 캐시에 저장
                    _vector_store_cache[cache_key] = vector_store
                    return vector_store
                else:
                    print("[빈 컬렉션 감지 - 삭제]")
                    client.delete_collection(collection_name)
            except:
                print("[컬렉션 상태 확인 실패 - 삭제 후 재생성]")
                try:
                    client.delete_collection(collection_name)
                except:
                    pass
    except:
        print("[컬렉션 목록 조회 실패]")
    
    # 새 컬렉션 생성 (필요한 경우만)
    print(f"[새 컬렉션 생성: {collection_name}]")
    
    try:
        # 문서 청킹 - 기존과 동일
        documents = split_chunks(file_path)
        if not documents:
            return None
        
        # 컬렉션 생성
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        
        # 벡터스토어 생성 및 문서 추가
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OllamaEmbeddings(model="bge-m3:567m")
        )
        
        print("임베딩 및 저장 중...")
        ids = [doc.metadata['order'] for doc in documents]
        vector_store.add_documents(documents, ids=ids)
        
        # 캐시에 저장
        _vector_store_cache[cache_key] = vector_store
        print(f"[벡터스토어 캐싱 완료: {len(documents)}개 문서]")
        
        return vector_store
        
    except Exception as e:
        print(f"[벡터스토어 생성 실패: {e}]")
        return None

def smart_determine_params(query: str):
    """개선된 파라미터 결정 - 답변 품질 고려"""

    
    # 복잡한 작업 (더 많은 토큰과 문서 필요)
    if any(keyword in query for keyword in ['보고서', '발표', 'ppt', '분석', '비교', '평가']):
        return 1000, 4096, "복합분석"
    
    # 퀴즈/문제 (적당한 양의 문서, 구조화된 답변)
    elif any(keyword in query for keyword in ['퀴즈']):
        return 100, 2048, "퀴즈"
    
    # 요약 (전체적인 이해 필요)
    elif any(keyword in query for keyword in ['요약', '정리', '핵심', '간추']):
        return 100, 1024, "요약"
    
    # 구체적 질문 (관련성 높은 문서 필요)
    elif any(keyword in query for keyword in ['어떻게', '왜', '무엇', '언제', '어디서', '누가']):
        return 100, 2048, "구체적질문"
    
    # 일반 질문
    else:
        return 10, 1024, "일반"

def create_enhanced_prompt(query: str, combined_text: str, history: str, task_type: str):
    """향상된 프롬프트 생성"""
    
    base_context = f"""
[LANGUAGE INSTRUCTION - MANDATORY]
**반드시 한국어로만 답변하세요. 중국어, 영어, 일본어 등 다른 언어는 절대 사용 금지입니다.**
**ONLY Korean language allowed. Chinese/English/Japanese strictly forbidden.**
**只能用韩语回答，严禁使用中文或其他语言。**

- 모든 답변은 반드시 한국어로만 작성해주세요
- 요약, 보고서 작성, 발표자료 작성에 특화되어있습니다
- 중국어나 영어가 포함된 답변은 절대 제공하지 마세요

이전 대화 기록:
{history}

참고 문서 내용:
{combined_text}

사용자 질문: {query}

**다시 한 번 강조: 답변은 100% 한국어로만 작성해주세요.**
"""

    if task_type == "복합분석":
        return f"""{base_context}

**한국어로만 답변 필수**
위 문서를 바탕으로 사용자의 요청에 대해 체계적이고 전문적으로 한국어로 답변해주세요:
- 반드시 한국어로만 답변합니다
- 문서의 핵심 내용을 충분히 반영하세요
- 논리적 구조로 답변을 구성하세요
- 구체적인 근거와 예시를 포함하세요
- 문서에 없는 내용은 추측하지 마세요
- 중국어/영어 사용 절대 금지

한국어 답변:"""

    elif task_type == "퀴즈":
        return f"""{base_context}

**한국어 퀴즈 생성**
문서 내용을 기반으로 한국어로만 퀴즈를 생성해주세요:
- 문서의 핵심 개념과 중요한 정보를 중심으로 구성하세요
- 다양한 유형의 문제를 포함하세요 (객관식, 단답형, 서술형 등)
- 사용자의 요청이 없다면 문제는 5개만 생성합니다
- 각 문제에 대한 정답과 해설을 제공하세요
- 난이도를 적절히 조절하세요
- 반드시 한국어로만 작성하세요

한국어 퀴즈:"""

    elif task_type == "요약":
        return f"""{base_context}

**한국어 요약**
문서의 주요 내용을 체계적으로 한국어로만 요약해주세요:
- 핵심 주제와 요점을 명확히 정리하세요
- 중요도에 따라 내용을 구조화하세요
- 구체적인 데이터나 예시가 있다면 포함하세요
- 간결하지만 포괄적으로 정리하세요
- 반드시 한국어로만 작성하세요

한국어 요약:"""

    elif task_type == "구체적질문":
        return f"""{base_context}

**한국어로 구체적 답변**
문서를 참조하여 구체적이고 정확하게 한국어로만 답변해주세요:
- 문서에서 관련된 정보를 찾아 근거로 제시하세요
- 단계별로 명확하게 설명하세요
- 문서에 명시되지 않은 부분은 "문서에서 확인할 수 없습니다"라고 명시하세요
- 가능한 한 구체적인 예시나 수치를 포함하세요
- 반드시 한국어로만 답변하세요

한국어 답변:"""

    else:  # 일반
        return f"""{base_context}

**한국어로 일반 답변**
문서를 바탕으로 사용자의 질문에 정확하고 친절하게 한국어로만 답변해주세요:
- 문서의 관련 내용을 충분히 활용하세요
- 명확하고 이해하기 쉽게 설명하세요
- 추가적인 맥락이나 배경 정보도 제공하세요
- 문서 범위를 벗어나는 추측은 피하세요
- 답변은 너무 길지 않게 해주세요
- 반드시 한국어로만 답변하세요

한국어 답변:"""

def question_answer_with_memory(file_path: str, query: str, memory: BufferMemory, tokens=256) -> str:
    """개선된 메인 함수 - 답변 품질과 성능 균형"""
    
    # 1. 향상된 파라미터 결정
    k, optimized_tokens, task_type = smart_determine_params(query)
    final_tokens = max(tokens, optimized_tokens) if tokens != 256 else optimized_tokens
    
    print(f"[작업 유형: {task_type}, 문서 수: {k}, 토큰: {final_tokens}]")
    
    # 2. 벡터스토어 로드 (캐싱됨)
    vector_store = data_to_vectorstore(file_path)
    
    # 3. 벡터스토어 실패 시 즉시 폴백
    if vector_store is None:
        print("[벡터스토어 없음 - 직접 파일 읽기]")
        return handle_fallback_mode(file_path, query, memory, final_tokens, task_type)
    
    # 4. 향상된 벡터 검색
    try:
        docs = vector_store.similarity_search(query, k=k)
        
        # 검색 결과가 부족한 경우 추가 검색
        if len(docs) < k//2:
            # 쿼리를 단순화해서 다시 검색
            simple_query = " ".join(query.split()[:3])  # 처음 3단어만
            additional_docs = vector_store.similarity_search(simple_query, k=k)
            # 중복 제거하면서 합치기
            seen = set()
            all_docs = []
            for doc in docs + additional_docs:
                doc_hash = hash(doc.page_content[:100])  # 첫 100자로 중복 판단
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    all_docs.append(doc)
                if len(all_docs) >= k:
                    break
            docs = all_docs
        
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 텍스트가 너무 짧은 경우 추가 문서 검색
        if len(combined_text) < 500:
            extra_docs = vector_store.similarity_search("", k=5)  # 일반적인 문서들
            for doc in extra_docs:
                if doc not in docs:
                    docs.append(doc)
                    combined_text += "\n\n" + doc.page_content
                if len(combined_text) > 1000:
                    break
        
    except Exception as e:
        print(f"[검색 실패: {e}] - 폴백 모드")
        return handle_fallback_mode(file_path, query, memory, final_tokens, task_type)
    
    # 5. 향상된 프롬프트로 LLM 호출
    history = memory.get_formatted_history()
    prompt = create_enhanced_prompt(query, combined_text, history, task_type)
    
    # 6. LLM 호출
    try:
        llm = get_llm(final_tokens)
        answer = llm.invoke(prompt).content
        
        # 한국어 응답 확인 및 처리
        answer = ensure_korean_only(answer)
        
        # 메모리 업데이트
        memory.append(query, answer)
        return answer
        
    except Exception as e:
        print(f"[LLM 실패: {e}] - 폴백 모드")
        return handle_fallback_mode(file_path, query, memory, final_tokens, task_type)

def handle_fallback_mode(file_path: str, query: str, memory: BufferMemory, tokens: int, task_type: str = "일반") -> str:
    """개선된 폴백 모드"""
    
    try:
        # 파일 크기에 따라 읽을 양 조절
        file_size = os.path.getsize(file_path)
        
        if file_size > 50000:  # 50KB 이상
            read_size = 15000  # 15KB만 읽기
        elif file_size > 20000:  # 20KB 이상
            read_size = 10000  # 10KB만 읽기
        else:
            read_size = file_size  # 전체 읽기
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(read_size)
        
        history = memory.get_formatted_history()
        
        # 폴백 모드에서도 한국어 강제 프롬프트 적용
        prompt = f"""
[LANGUAGE INSTRUCTION - MANDATORY]
**반드시 한국어로만 답변하세요. 중국어, 영어, 일본어 등 다른 언어는 절대 사용 금지입니다.**

당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다. 
모든 답변은 반드시 한국어로만 작성해주세요.

이전 대화 기록:
{history}

참고 문서 내용:
{content}

사용자 질문: {query}

**한국어로만 답변:**
"""
        
        # 토큰 수 조절 (폴백 모드에서는 약간 줄임)
        fallback_tokens = min(tokens, 2048)
        
        llm = get_llm(fallback_tokens)
        answer = llm.invoke(prompt).content
        
        # 폴백 모드에서도 한국어 응답 확인
        answer = ensure_korean_only(answer)
        
        memory.append(query, answer)
        return answer
        
    except Exception as e:
        return f"죄송합니다. 문서 처리 중 오류가 발생했습니다: {str(e)}\n\n다시 시도해 주시거나 문서 형식을 확인해 주세요."

def translate_to_korean(text: str) -> str:
    """중국어나 영어 텍스트를 한국어로 번역"""
    try:
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.1,
            timeout=30.0
        )
        
        # 번역 전용 프롬프트
        translation_prompt = f"""
당신은 전문 번역가입니다. 주어진 텍스트를 정확하고 자연스러운 한국어로 번역해주세요.

번역 규칙:
1. 원문의 의미와 뉘앙스를 정확히 보존할 것
2. 자연스럽고 읽기 쉬운 한국어로 번역할 것
3. 전문 용어는 적절한 한국어 용어로 번역할 것
4. 문단 구조와 서식을 유지할 것
5. 번역문만 출력하고 추가 설명은 하지 말 것

번역할 텍스트:
{text}

한국어 번역:"""

        response = llm.invoke(translation_prompt)
        
        if response and hasattr(response, 'content'):
            translated = response.content.strip()
            
            # 번역 결과가 유효한지 확인 (한국어 포함 여부)
            korean_chars = len(re.findall(r'[가-힣]', translated))
            if korean_chars > 0:
                return translated
        
        return None
        
    except Exception as e:
        print(f"[번역 오류] {str(e)}")
        return None

def ensure_korean_only(text: str) -> str:
    """중국어 중심의 응답을 한국어로 번역하여 반환"""
    if not text or not isinstance(text, str):
        return "죄송합니다. 응답을 생성할 수 없습니다."
    
    # 한국어 문자 비율 체크
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'[\s\n\r\t\.,;:!?\-\(\)\[\]{}\"\'`~@#$%^&*+=|\\/<>]', '', text))
    
    # 텍스트가 너무 짧으면 그대로 통과
    if total_chars < 10:
        return text
    
    # 한국어가 전혀 없고 중국어가 많은 경우 번역 시도
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))
    
    # 중국어 응답 감지: 한국어가 10% 미만이면서 중국어가 30% 이상인 경우
    if korean_chars < total_chars * 0.1 and chinese_chars > total_chars * 0.3:
        print("[언어 감지] 중국어 응답 감지됨, 한국어로 번역 시도 중...")
        translated = translate_to_korean(text)
        if translated:
            return f"{translated}\n\n💡 원본이 중국어로 생성되어 한국어로 번역하였습니다."
        return "죄송합니다. 한국어로만 답변드릴 수 있습니다. 다시 질문해주세요."
    
    # 완전 영어 응답 감지: 한국어가 5% 미만이면서 연속된 영어 문장이 많은 경우
    english_sentences = re.findall(r'\b[A-Za-z]+(?:\s+[A-Za-z]+){4,}\b', text)
    if korean_chars < total_chars * 0.05 and len(' '.join(english_sentences)) > total_chars * 0.5:
        print("[언어 감지] 영어 응답 감지됨, 한국어로 번역 시도 중...")
        translated = translate_to_korean(text)
        if translated:
            return f"{translated}\n\n💡 원본이 영어로 생성되어 한국어로 번역하였습니다."
        return "죄송합니다. 한국어로만 답변드릴 수 있습니다. 다시 질문해주세요."
    
    # 정상적인 한국어 응답은 그대로 반환
    return text

def clear_cache():
    """캐시 초기화"""
    global _vector_store_cache, _client_cache
    _vector_store_cache.clear()
    _client_cache = None
    print("[캐시 초기화 완료]")

def get_cache_stats():
    """캐시 상태 확인"""
    return {
        "vector_stores": len(_vector_store_cache),
        "client_connected": _client_cache is not None
    }

# Django 호환성 유지
if __name__ == "__main__":
    # 테스트 코드
    memory = BufferMemory()
    
    print("=== 개선된 버전 테스트 ===")
    
    import time
    start_time = time.time()
    result1 = question_answer_with_memory("temp/sample.txt", "이 문서의 주요 내용은 무엇인가요?", memory)
    print(f"첫 번째 답변: {result1[:200]}...")
    
    start_time = time.time()
    result2 = question_answer_with_memory("temp/sample.txt", "구체적으로 어떤 기술이 사용되었나요?", memory)
    print(f"두 번째 답변: {result2[:200]}...")
    
    print(f"\n캐시 상태: {get_cache_stats()}")