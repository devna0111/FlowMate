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
# from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from vectordb_upload_search import data_to_vectorstore, question_answer_with_memory, BufferMemory
from utils.docx_writer import markdown_to_styled_docx
from utils.pptx_writer import save_structured_text_to_pptx
from utils.intent_classifier import check_intent, normalize_label
from utils.eval_hr import hr_predict

# 업로드/생성 파일 폴더 경로 분리
TEMP_DIR = os.path.join(settings.BASE_DIR, "temp")
UPLOADS_DIR = os.path.join(settings.BASE_DIR, "uploads")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

DEFAULT_FILE_PATH = os.path.join(TEMP_DIR, "sample.txt")

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
    """보고서/발표자료 등 생성 파일은 uploads/ 폴더에 저장"""
    llm = ChatOllama(model='qwen2.5:7b-instruct')
    
    if request.method == "POST":
        # 세션 미생성 시 강제 생성
        if not request.session.session_key:
            request.session.create()

        data = json.loads(request.body)
        query = data.get("message")
        file_path = data.get("file_path", DEFAULT_FILE_PATH)

        # 세션에서 BufferMemory 불러오기
        memory = get_buffer_memory_from_session(request.session)

        # report_keywords = ['보고서', '보고서 작성', '보고서 초안', '보고서 생성']
        # pptx_keywords = ['발표문', '발표 자료', '발표자료', '발표 초안', '발표 자료 초안']
        # summary_keywords = ['요약해줘', '요약본', '3줄요약', '요약문', '핵심만',"정리해줘",'간추려줘']
        
        check_intents = check_intent(query)
        result = llm.invoke(check_intents.format_messages(query=query))
        intent = normalize_label(result.content)
        print(intent)
        # if any(k in query for k in report_keywords):
        if "[보고서]" in intent :
            prompt = (
                '당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다. 아래의 내용에 한국어로만 친절히 답변해주세요.\n'
                "아래 문서 내용을 기반으로 다음 조건을 만족하는 보고서를 마크다운(Markdown) 형식으로 작성해줘.\n"
                "0. 반복되는 말을 하지 않는다.\n"
                "1. 문서의 주제와 목적, 주요 내용, 결론, 권고사항을 명확히 포함할 것.\n"
                "2. 목차, 표, 리스트, 강조문구 등은 문서 내용을 반영하여 구성할 것.\n"
                "3. 과도한 추론이나 창작은 하지 말고, 문서에 없는 정보는 생성하지 마라.\n"
                "4. 어떤 종류의 문서든 적절한 구조의 보고서 초안이 되도록 작성할 것.\n"
                "5. 표지(제목), 목차, 본문(각 항목별 요약/분석/핵심 내용), 결론 및 권고사항 순으로 작성.\n"
                "사용자 요청: "
                + query
            )
            markdown = question_answer_with_memory(file_path, prompt, memory, tokens=4096).strip()

            unique_id = uuid.uuid4().hex
            docx_name = "report_sample.docx"
            docx_path = os.path.join(UPLOADS_DIR, docx_name)
            markdown_to_styled_docx(markdown, output_path=docx_path)

            # BufferMemory를 세션에 저장
            save_buffer_memory_to_session(request.session, memory)

            return JsonResponse({
                "report_markdown": markdown,
                "report_file_url": f"/download_report/?filename={docx_name}"
            })

        # elif any(k in query for k in pptx_keywords):
        elif "[발표]" in intent :
            prompt = f"""
                    [user] {query}
                    [system]
                    당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다. 아래의 내용에 한국어로 친절히 답변해주세요.
                    
                    당신은 AI 발표자료 자동화 전문가입니다.
                    아래 [발표자료]의 내용을 바탕으로 실제 PPT 슬라이드를 만들기 위한 "슬라이드 구성 텍스트"를 한국어로만 작성해주세요.

                    조건:
                    1. 전체 내용을 5~10개 슬라이드로 논리적으로 나누세요.
                    2. 각 슬라이드는 반드시 아래와 같이 작성합니다:

                    [슬라이드 N]
                    제목: (슬라이드 제목, 발표 흐름에 맞게)
                    핵심 포인트:
                    - (핵심 메시지1, 문서 내용 기반)
                    - (핵심 메시지2)
                    - (필요하면 추가)

                    3. 슬라이드 표지(제목/프로젝트 명)와 마지막 슬라이드(결론/요약/권고)는 꼭 포함하세요.
                    4. 발표 흐름(도입 → 목적/배경 → 주요 내용/분석 → 결론/권고)에 따라 슬라이드를 구성하세요.
                    5. 각 슬라이드의 '제목'과 '핵심 포인트'는 원문 내용에서 최대한 추출하고, 없는 내용은 과도하게 상상/창작하지 마세요.
                    6. 불필요한 서문, 해설, 기타 설명은 절대 작성하지 말고, 반드시 [슬라이드 N] ~ 형식만 반복하세요.
                    7. 반복되는 말은 하지 마세요.
                    8. 답변은 한국어로만 합니다

                    [발표자료]
                    """ 
            markdown = question_answer_with_memory(file_path, prompt, memory, tokens=4096).strip()

            unique_id = uuid.uuid4().hex
            pptx_name = f"pptx_sample.pptx"
            pptx_path = os.path.join(UPLOADS_DIR, pptx_name)
            save_structured_text_to_pptx(markdown, output_path=pptx_path)

            save_buffer_memory_to_session(request.session, memory)

            return JsonResponse({
                "report_markdown": markdown,
                "report_file_url": f"/download_report/?filename={pptx_name}"
            })
            
        # elif any(k in query for k in summary_keywords):
        elif "[요약]" in intent :
            prompt = f"""
                    [system]
                    당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다. 아래의 내용에 한국어로 친절히 답변해주세요.
                    당신은 요약을 도와주는 어시스턴트입니다.
                    한국어로 답변합니다.
                    
                    - 구체적인 출처 경로 또는 예시를 제시하고
                    - 문장 구조를 명확히 하며
                    - 실무적으로 바로 활용할 수 있도록
                    - 반복되는 말은 하지 않고
                    
                    한국어로 정확하고 친절하게 답변해주세요.
                    [user] {query}
                    """ 
            answer = question_answer_with_memory(file_path, query, memory, tokens=1024)

            save_buffer_memory_to_session(request.session, memory)
            return JsonResponse({"answer": answer})

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