import os
import hashlib
from parsing_utils import split_chunks
from langchain_qdrant import Qdrant
from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from collections import deque

class BufferMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def append(self, user, assistant):
        self.history.append({"user": user, "assistant": assistant})

    def get_formatted_history(self):
        # LLM에 넣을 때 사용
        return "\n".join(
            [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history]
        )

def get_file_hash(file_path: str) -> str:
    """파일의 해시값을 기반으로 고유 컬렉션 이름 생성"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def data_to_vectorstore(file_path: str):
    # Qdrant 클라이언트 연결
    client = QdrantClient(host="localhost", port=6333)
    print("[Qdrant 연결 성공]")

    # 파일 기반 고유 컬렉션 이름 생성
    collection_name = f"doc_{get_file_hash(file_path)}"

    # 이미 존재하는 컬렉션인지 확인
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name in existing_collections:
        print(f"[이미 존재하는 컬렉션: {collection_name}] → 기존 DB 사용")
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OllamaEmbeddings(model="bge-m3:567m")
        )
    documents = split_chunks(file_path)
    # 컬렉션 새로 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"[새 컬렉션 생성: {collection_name}]")

    # 문서 임베딩
    
    embedding_function = OllamaEmbeddings(model="bge-m3:567m")
    print("bge-m3 준비 완료")
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_function
    )
    print("임베딩 하고 DB 적재")
    qdrant.add_documents(documents)
    print(f"[문서 {len(documents)}개 벡터 저장 완료]")

    return qdrant

def question_answer_with_memory(file_path: str, query: str, memory: BufferMemory) -> str:
    vector_store = data_to_vectorstore(file_path)

    # 중요 키워드일 경우 전체 검색 확대
    filter_keywords = ['요약', '발표', '퀴즈', '보고서']
    k = 1000 if any(f in query for f in filter_keywords) else 5
    tokens = 2048 if any(f in query for f in filter_keywords) else 256

    docs = vector_store.similarity_search(query, k=k)
    combined_text = "\n\n".join([doc.page_content for doc in docs])

    history = memory.get_formatted_history()

    prompt = f"""당신은 뛰어난 업무 보조입니다.
다음의 [질문사항]에 대해 [참고자료]와 [이전 대화 이력]을 바탕으로 정확하고 간결하게 답변하세요.

[이전 대화 이력]
{history}

[질문사항] 
{query}

[참고자료]
{combined_text}
"""

    llm = ChatOllama(model='qwen2.5vl:7b', temperature=0.2, repeat_penalty=1.15, max_token=tokens)
    answer = llm.invoke(prompt).content

    # 히스토리에 현재 대화 추가
    memory.append(query, answer)

    return answer

if __name__ == "__main__":
    file_path = "sample_inputs/sample.pdf"
    memory = BufferMemory(max_turns=5)

    while True:
        query = input("질문: ")
        if query in ["끝", "종료"]:
            break
        answer = question_answer_with_memory(file_path, query, memory)
        print("응답:", answer)
