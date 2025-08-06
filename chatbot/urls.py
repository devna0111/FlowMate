from django.urls import path, include
from .views import chat_page, upload_file, ask_question, list_uploaded_files, home, download_report, list_generated_files


urlpatterns = [
    path("", home, name="home"),
    path("chat_page/", chat_page, name="chat_page"),
    path("upload/", upload_file, name="upload_file"),
    path("ask/", ask_question, name="ask_question"),
    path("list_files/", list_uploaded_files, name="list_files"),
    path("list_generated_files/", list_generated_files, name="list_generated_files"),  # ← 추가!
    path("presentation/", include('presentation.urls')),
    path("download_report/", download_report, name="download_report"),
]
