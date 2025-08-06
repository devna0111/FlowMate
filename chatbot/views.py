from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from langchain_ollama import ChatOllama
import os
from django.conf import settings
from vectordb_upload_search import data_to_vectorstore, question_answer_with_memory, BufferMemory

llm = ChatOllama(model = 'qwen2.5vl:7b')
# Create your views here.

def home(request) :
    return render(request, "home.html")

def generate_llm_answer(message: str) -> str:
    # Ollama 로컬에서 실행 중인 모델 사용
    llm = ChatOllama(model="qwen2.5vl:7b", max_tokens=512)
    prompt = "가능한 간결하게 답변하세요."
    response = llm.invoke(prompt + message)
    return response.content.strip()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "../temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def chat_page(request):
    return render(request, "chatbot/chat.html")

@csrf_exempt
def upload_file(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return JsonResponse({"success": False, "message": "파일이 없습니다."})

        # 파일 저장
        save_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # 벡터화 진행
        try:
            data_to_vectorstore(save_path)
        except Exception as e:
            return JsonResponse({"success": False, "message": f"벡터화 실패: {str(e)}"})

        return JsonResponse({"success": True, "file_path": save_path})
    return JsonResponse({"success": False, "message": "POST 요청만 지원합니다."})

user_memory = {}
DEFAULT_FILE_PATH = "temp/sample.txt"
@csrf_exempt
def ask_question(request):
    import json
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("message")
        file_path = data.get("file_path", DEFAULT_FILE_PATH)

        # 사용자 세션 기반 메모리 유지
        session_id = request.session.session_key or "default"
        if session_id not in user_memory:
            user_memory[session_id] = BufferMemory()

        memory = user_memory[session_id]
        answer = question_answer_with_memory(file_path, query, memory)

        return JsonResponse({"answer": answer})

TEMP_DIR = os.path.join(settings.BASE_DIR, "temp")

def list_uploaded_files(request):
    try:
        files = os.listdir(TEMP_DIR)
        files = [f for f in files if not f.startswith('.')]  # 숨김 파일 제외
        return JsonResponse({"files": files})
    except Exception as e:
        return JsonResponse({"files": [], "error": str(e)})