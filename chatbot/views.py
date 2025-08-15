from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import os, json, uuid
from django.conf import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb_upload_search import data_to_vectorstore, question_answer_with_memory, BufferMemory, ensure_korean_only
from utils.docx_writer import markdown_to_styled_docx
from utils.pptx_writer import save_structured_text_to_pptx
from utils.intent_classifier import check_intent, normalize_label
from utils.eval_hr import hr_predict
from langgraph_workflow import execute_workflow, TaskType

# 업로드/생성 파일 폴더 경로 분리
TEMP_DIR = os.path.join(settings.BASE_DIR, "temp")
UPLOADS_DIR = os.path.join(settings.BASE_DIR, "uploads")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

DEFAULT_FILE_PATH = os.path.join(TEMP_DIR, "sample.txt")

def create_korean_only_prompt_template():
    """한국어 강제 프롬프트 템플릿 생성"""
    return ChatPromptTemplate.from_messages([
        ("system", 
        "[CRITICAL LANGUAGE INSTRUCTION]"
        "반드시 한국어로만 답변하세요. 중국어, 영어, 일본어 등 다른 언어는 절대 사용 금지입니다. "
        "ONLY Korean language allowed. Chinese/English/Japanese strictly forbidden. "
        "只能用韩语回答，严禁使用中文或其他语言。 "
        "입력 텍스트에서 한국어가 아닌 부분은 한국어로 번역하고, "
        "원래 문서의 구조·양식(헤더/리스트/표 등)은 최대한 그대로 유지하여 반환하세요. "
        "100% 한국어 출력만 허용됩니다."),
        ("user", "{input}")
    ])

def create_report_prompt(query):
    """보고서 생성 프롬프트"""
    return (
        '당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다. 아래의 내용에 한국어로만 친절히 답변해주세요.\n'
        "아래 문서 내용을 기반으로 다음 조건을 만족하는 보고서를 마크다운(Markdown) 형식으로 작성해줘.\n"
        "0. 반복되는 말을 하지 않는다.\n"
        "1. 문서의 주제와 목적, 주요 내용, 결론, 권고사항을 명확히 포함할 것.\n"
        "2. 목차, 표, 리스트, 강조문구 등은 문서 내용을 반영하여 구성할 것.\n"
        "3. 과도한 추론이나 창작은 하지 말고, 문서에 없는 정보는 생성하지 마라.\n"
        "4. 어떤 종류의 문서든 적절한 구조의 보고서 초안이 되도록 작성할 것.\n"
        "5. 표지(제목), 목차, 본문(각 항목별 요약/분석/핵심 내용), 결론 및 권고사항 순으로 작성.\n"
        "사용자 요청: " + query
    )

def create_presentation_prompt(query):
    """발표자료 생성 프롬프트"""
    return (
        '당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다. 아래의 내용에 한국어로만 친절히 답변해주세요.\n'
        "아래 문서내용을 바탕으로 실제 PPT 슬라이드를 만들기 위한 '슬라이드 구성 텍스트'를 한국어로만 작성해줘.\n"
        "0. 반복되는 말을 하지 않는다.\n"
        "1. 전체 내용을 5~10개 슬라이드로 논리적으로 나눈다.\n"
        "2. 각 슬라이드는 반드시 아래와 같이 작성.\n"
        """출력 형식:
            [슬라이드 1]
            제목: 프로젝트 소개
            핵심 포인트:
            - 첫 번째 포인트
            - 두 번째 포인트"""
        "3. 슬라이드 표지(제목/프로젝트 명)와 마지막 슬라이드(결론/요약/권고)는 꼭 포함.\n"
        "4. 어떤 종류의 문서든 적절한 구조의 보고서 초안이 되도록 작성할 것.\n"
        "5. 발표 흐름(도입 → 목적/배경 → 주요 내용/분석 → 결론/권고)에 따라 슬라이드를 구성.\n"
        "6. 각 슬라이드의 '제목'과 '핵심 포인트'는 원문 내용에서 최대한 추출하고, 없는 내용은 상상/창작하지 말 것.\n"
        "7. [슬라이드 N] ~ 형식만 반복.\n"
        "8. 반복되는 말은 하지 마세요.\n"
        "사용자 요청: " + query
    )

# BufferMemory와 세션 연동 헬퍼 함수 추가!
def get_buffer_memory_from_session(session):
    """세션에서 BufferMemory 인스턴스를 불러오거나 새로 생성"""
    if "chat_history" not in session:
        session["chat_history"] = []  # 리스트로만 저장
    memory = BufferMemory()
    memory.history = session["chat_history"]  # 리스트 복원
    return memory

def save_buffer_memory_to_session(session, memory):
    """BufferMemory 인스턴스의 history를 세션에 저장"""
    session["chat_history"] = list(memory.history)
    session.modified = True

def home(request):
    context = {
        'user': request.user,
    }
    return render(request, "index.html", context)

@login_required
def chat_page(request):
    context = {
        'user': request.user,
    }
    return render(request, "chatbot/chat.html", context)

@csrf_exempt
def upload_file(request):
    """사용자 파일 업로드: temp/ 폴더에 저장"""
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return JsonResponse({"success": False, "message": "파일이 없습니다."})

        save_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        try:
            data_to_vectorstore(save_path)
        except Exception as e:
            return JsonResponse({"success": False, "message": f"벡터화 실패: {str(e)}"})
        return JsonResponse({"success": True, "file_path": save_path})
    return JsonResponse({"success": False, "message": "POST 요청만 지원합니다."})

@csrf_exempt
def ask_question(request):
    """LangGraph 워크플로우 기반 통합 처리"""
    if request.method == "POST":
        # 세션 미생성 시 강제 생성
        if not request.session.session_key:
            request.session.create()

        data = json.loads(request.body)
        query = data.get("message")
        file_path = data.get("file_path", DEFAULT_FILE_PATH)

        # 세션에서 BufferMemory 불러오기
        memory = get_buffer_memory_from_session(request.session)

        try:
            # LangGraph 워크플로우 실행
            result = execute_workflow(query, file_path, memory)
            
            # 결과 처리
            if result.success:
                # BufferMemory를 세션에 저장
                save_buffer_memory_to_session(request.session, memory)
                
                # 태스크별 응답 형식
                if result.task_type in [TaskType.REPORT, TaskType.PRESENTATION]:
                    filename = result.output_file_path.split('/')[-1] if result.output_file_path else None
                    return JsonResponse({
                        "report_markdown": result.final_response,
                        "report_file_url": f"/download_report/?filename={filename}" if filename else None
                    })
                else:
                    # 요약, 질의응답 등
                    return JsonResponse({
                        "answer": result.final_response
                    })
            else:
                # 에러 처리
                return JsonResponse({
                    "answer": result.final_response,
                    "error": result.error_message
                })
                
        except Exception as e:
            # 폴백: 기존 시스템 사용
            print(f"[LangGraph 워크플로우 실패] {str(e)} - 기존 시스템으로 폴백")
            answer = question_answer_with_memory(file_path, query, memory)
            save_buffer_memory_to_session(request.session, memory)
            return JsonResponse({"answer": answer})

    return JsonResponse({"error": "Invalid method"}, status=405)

def download_report(request):
    """생성 파일 다운로드 (uploads/만 접근)"""
    filename = request.GET.get("filename")
    if not filename:
        return JsonResponse({"error": "No filename provided"}, status=400)
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        return JsonResponse({"error": "File not found"}, status=404)
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)

def list_uploaded_files(request):
    """업로드 파일 리스트: temp/ 폴더만"""
    try:
        files = os.listdir(TEMP_DIR)
        files = [f for f in files if not f.startswith('.')]
        return JsonResponse({"files": files})
    except Exception as e:
        return JsonResponse({"files": [], "error": str(e)})

def list_generated_files(request):
    """생성파일 리스트: uploads/ 폴더만"""
    try:
        files = os.listdir(UPLOADS_DIR)
        files = [f for f in files if not f.startswith('.')]
        return JsonResponse({"files": files})
    except Exception as e:
        return JsonResponse({"files": [], "error": str(e)})

@csrf_exempt
def clear_history(request):
    """세션의 대화 히스토리를 초기화(삭제)"""
    if "chat_history" in request.session:
        del request.session["chat_history"]
        request.session.modified = True
    return JsonResponse({"cleared": True})

# 인증 관련 뷰들
def user_login(request):
    """사용자 로그인"""
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'{username}님, 환영합니다!')
                return redirect('home')
        messages.error(request, '잘못된 사용자명 또는 비밀번호입니다.')
    
    # GET 요청이거나 로그인 실패시
    return render(request, 'registration/login.html')

def signup(request):
    """사용자 회원가입"""
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'{username}님, 회원가입이 완료되었습니다!')
            login(request, user)
            return redirect('home')
        messages.error(request, '회원가입 중 오류가 발생했습니다.')
    
    # GET 요청이거나 회원가입 실패시
    return render(request, 'registration/signup.html')

def user_logout(request):
    """사용자 로그아웃"""
    logout(request)
    messages.info(request, '로그아웃되었습니다.')
    return redirect('home')

@login_required
def hr_evaluation_page(request):
    """HR 업무평가 예측 페이지"""
    return render(request, 'chatbot/hr_evaluation.html')

@csrf_exempt
@login_required
def hr_evaluation_predict(request):
    """HR 업무평가 예측 API"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            # 입력 데이터 추출
            age = int(data.get('age'))
            project = int(data.get('project'))
            salary = int(data.get('salary'))
            number_of_turnovers = int(data.get('number_of_turnovers'))
            surround_eval = int(data.get('surround_eval'))
            personal_history = int(data.get('personal_history'))
            edu_trips_lastyear = int(data.get('edu_trips_lastyear'))
            currentyear_at_company = int(data.get('currentyear_at_company'))
            buisnesstrip = int(data.get('buisnesstrip'))
            when_enroll = int(data.get('when_enroll'))
            
            # HR 평가 예측 수행
            result = hr_predict(
                age=age,
                project=project,
                salary=salary,
                number_of_turnovers=number_of_turnovers,
                surround_eval=surround_eval,
                personal_history=personal_history,
                edu_trips_lastyear=edu_trips_lastyear,
                currentyear_at_company=currentyear_at_company,
                buisnesstrip=buisnesstrip,
                when_enroll=when_enroll
            )
            
            return JsonResponse({
                "success": True,
                "result": result,
                "message": f"예측 결과: {result}"
            })
            
        except Exception as e:
            return JsonResponse({
                "success": False,
                "error": str(e),
                "message": "예측 중 오류가 발생했습니다."
            })
    
    return JsonResponse({"error": "POST 요청만 지원합니다."}, status=405)