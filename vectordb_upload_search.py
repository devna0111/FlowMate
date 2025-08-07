import os
import hashlib
from parsing_utils import split_chunks
# 최소한의 import로 시작 시간 단축
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

# 전역 캐시로 성능 향상 (LLM 캐시 제거)
_vector_store_cache = {}
_client_cache = None

class BufferMemory:
    """기존 BufferMemory - 성능 최적화"""
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
    """파일 해시 - 캐싱으로 최적화"""
    try:
        # 파일 수정 시간과 크기로 간단한 해시 생성 (더 빠름)
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
    """LLM 캐싱 - max_tokens 호환성 문제 해결"""
    global _llm_cache
    
    # 매번 새로 생성 (토큰 수 때문에 캐싱 효과가 제한적이므로)
    try:
        # 최신 버전에서는 num_predict 사용
        return ChatOllama(
            model='qwen2.5vl:7b', 
            temperature=0.2, 
            repeat_penalty=1.15, 
            num_predict=tokens
        )
    except TypeError:
        try:
            # 구버전에서는 max_tokens 사용
            return ChatOllama(
                model='qwen2.5vl:7b', 
                temperature=0.2, 
                repeat_penalty=1.15, 
                max_tokens=tokens
            )
        except TypeError:
            # 둘 다 안 되면 기본 설정만
            return ChatOllama(
                model='qwen2.5vl:7b', 
                temperature=0.2, 
                repeat_penalty=1.15
            )

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
                if collection_info.vectors_count > 0:
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
        vector_store.add_documents(documents)
        
        # 캐시에 저장
        _vector_store_cache[cache_key] = vector_store
        print(f"[벡터스토어 캐싱 완료: {len(documents)}개 문서]")
        
        return vector_store
        
    except Exception as e:
        print(f"[벡터스토어 생성 실패: {e}]")
        return None

def fast_determine_params(query: str):
    """빠른 파라미터 결정 (복잡한 키워드 매칭 제거)"""
    query_lower = query.lower()
    
    if '보고서' in query or '발표' in query or 'ppt' in query_lower:
        return 50, 4096, "보고서/발표"
    elif '퀴즈' in query or '문제' in query or '테스트' in query:
        return 100, 3072, "퀴즈"
    elif '요약' in query or '정리' in query:
        return 20, 1024, "요약"  # 토큰 수 줄임
    else:
        return 5, 512, "일반"

def question_answer_with_memory(file_path: str, query: str, memory: BufferMemory, tokens=256) -> str:
    """최적화된 메인 함수"""
    
    # 1. 빠른 파라미터 결정
    k, optimized_tokens, task_type = fast_determine_params(query)
    final_tokens = max(tokens, optimized_tokens) if tokens != 256 else optimized_tokens
    
    # 2. 벡터스토어 로드 (캐싱됨)
    vector_store = data_to_vectorstore(file_path)
    
    # 3. 벡터스토어 실패 시 즉시 폴백 (빠른 경로)
    if vector_store is None:
        print("[벡터스토어 없음 - 직접 파일 읽기]")
        return handle_fallback_mode(file_path, query, memory, final_tokens)
    
    # 4. 벡터 검색
    try:
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            # 빈 검색 시 전체 검색 (간단하게)
            docs = vector_store.similarity_search("", k=min(10, k))
        
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
    except Exception as e:
        print(f"[검색 실패: {e}] - 폴백 모드")
        return handle_fallback_mode(file_path, query, memory, final_tokens)
    
    # 5. 간소화된 프롬프트로 LLM 호출
    history = memory.get_formatted_history()
    
    # 단순화된 프롬프트 (성능상 이유)
    if task_type == "보고서/발표":
        prompt = f"""전문 보고서/발표자료를 작성하세요.

대화기록: {history}

요청: {query}

문서: {combined_text}

지침: 문서 기반으로 체계적이고 전문적으로 작성하세요."""

    elif task_type == "퀴즈":
        prompt = f"""문서 기반 퀴즈를 만드세요.

대화기록: {history}

요청: {query}

문서: {combined_text}

지침: 핵심 개념 중심의 다양한 문제와 해설을 포함하세요."""

    else:
        # 일반 질문/요약
        prompt = f"""문서를 바탕으로 정확히 답변하세요.

대화기록: {history}

질문: {query}

문서: {combined_text}

답변:"""
    
    # 6. LLM 호출 (캐싱된 모델 사용)
    try:
        llm = get_llm(final_tokens)
        answer = llm.invoke(prompt).content
        
        # 메모리 업데이트
        memory.append(query, answer)
        return answer
        
    except Exception as e:
        print(f"[LLM 실패: {e}] - 폴백 모드")
        return handle_fallback_mode(file_path, query, memory, final_tokens)

def handle_fallback_mode(file_path: str, query: str, memory: BufferMemory, tokens: int) -> str:
    """폴백 모드 - 최소 기능으로 빠르게"""
    
    try:
        # 파일 직접 읽기 (빠른 처리를 위해 크기 제한)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()[:5000]  # 첫 5000자만 (빠른 처리)
        
        history = memory.get_formatted_history()
        
        # 최소 프롬프트
        simple_prompt = f"""문서 기반 답변:

대화: {history}
질문: {query}
문서: {content}

답변:"""
        
        llm = get_llm(min(tokens, 1024))  # 토큰 제한으로 빠르게
        answer = llm.invoke(simple_prompt).content
        
        memory.append(query, answer)
        return answer
        
    except Exception as e:
        return f"처리 중 오류가 발생했습니다: {str(e)}"

def clear_cache():
    """캐시 초기화 (메모리 절약 필요시)"""
    global _vector_store_cache, _client_cache
    _vector_store_cache.clear()
    _client_cache = None
    print("[캐시 초기화 완료]")

# Django 호환성 유지
if __name__ == "__main__":
    # 간단한 테스트
    memory = BufferMemory()
    
    print("=== 성능 최적화 버전 테스트 ===")
    
    # 첫 번째 호출 (벡터스토어 생성)
    import time
    start_time = time.time()
    result1 = question_answer_with_memory("temp/sample.txt", "이 문서는 뭐에 관한 거야?", memory)
    first_call_time = time.time() - start_time
    
    # 두 번째 호출 (캐시 활용)
    start_time = time.time()
    result2 = question_answer_with_memory("temp/sample.txt", "주요 내용 요약해줘", memory)
    second_call_time = time.time() - start_time
    
    print(f"\n첫 번째 호출 시간: {first_call_time:.2f}초")
    print(f"두 번째 호출 시간: {second_call_time:.2f}초")
    print(f"속도 향상: {first_call_time/second_call_time:.1f}배")
    print(f"\n캐시 상태: {get_cache_stats()}")