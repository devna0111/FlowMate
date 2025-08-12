from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from utils.run_feedback_pipeline import run_feedback_pipeline
from django.contrib.auth.decorators import login_required
import traceback
import os

@login_required
def presentation(request):
    return render(request, 'presentation/analysis.html')

def upload_video(request):
    """
    영상 파일 업로드를 처리하는 API 뷰입니다.
    프론트엔드에서 영상 파일을 전송하면, 서버에 저장 후 저장 경로를 반환합니다.
    """
    if request.method == "POST" and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # 저장 폴더 생성 (없으면 자동 생성)
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # 파일 이름 중복 방지 (간단 버전: 같은 이름 있으면 뒤에 숫자 추가)
        save_path = os.path.join(upload_dir, video_file.name)
        base, ext = os.path.splitext(save_path)
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base}_{counter}{ext}"
            counter += 1

        # 실제 파일 저장
        with open(save_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        # 프론트엔드에 저장 경로 전달 (실제 사용 시 보안상 파일명만 주는 게 더 안전)
        return JsonResponse({
            "success": True,
            "filename": os.path.basename(save_path),
            "video_path": save_path  # 필요에 따라 실제 경로 대신 상대 경로만 전달 권장
        })
    else:
        return JsonResponse({
            "success": False,
            "error": "No video file uploaded"
        }, status=400)

@csrf_exempt  # (개발 단계에서만. 실제 서비스 시 CSRF 처리 필수!)
def analyze_video(request):
    """
    업로드된 영상을 분석 파이프라인에 전달하고, 결과를 JSON으로 반환합니다.
    """
    if request.method == "POST":
        import json
        try:
            body = json.loads(request.body)
            video_path = body.get('video_path')

            if not video_path or not os.path.exists(video_path):
                return JsonResponse({"success": False, "error": "Invalid video path."}, status=400)
            
            result = run_feedback_pipeline(video_path)
            return JsonResponse({
                "success": True,
                **result
            })
        except Exception as e :
            return JsonResponse({
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc()
            }, status=500)
    else:
        return JsonResponse({"success": False, "error": "Invalid request"}, status=400)