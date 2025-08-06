from django.urls import path
from presentation import views

urlpatterns = [
    path('', views.presentation, name='presentation'),
    path('upload/', views.upload_video, name='upload_video'),  # 업로드 API 추가
    path('analyze/', views.analyze_video, name='analyze_video'),  # 분석 API 추가
]