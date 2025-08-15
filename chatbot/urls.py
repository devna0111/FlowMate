from django.urls import path, include
from .views import chat_page, upload_file, ask_question, list_uploaded_files, home, download_report, list_generated_files, clear_history, user_login, signup, user_logout, hr_evaluation_page, hr_evaluation_predict


urlpatterns = [
    path("", home, name="home"),
    path("chat_page/", chat_page, name="chat_page"),
    path("upload/", upload_file, name="upload_file"),
    path("ask/", ask_question, name="ask_question"),
    path("list_files/", list_uploaded_files, name="list_files"),
    path("list_generated_files/", list_generated_files, name="list_generated_files"),  # ← 추가!
    path("presentation/", include('presentation.urls')),
    path("download_report/", download_report, name="download_report"),
    path("clear_history/", clear_history, name="clear_history"),
    
    # 인증 관련 URL
    path("login/", user_login, name="user_login"),
    path("signup/", signup, name="signup"),
    path("logout/", user_logout, name="user_logout"),
    
    # HR 평가 관련 URL
    path("hr_evaluation/", hr_evaluation_page, name="hr_evaluation_page"),
    path("hr_evaluation/predict/", hr_evaluation_predict, name="hr_evaluation_predict"),
]
