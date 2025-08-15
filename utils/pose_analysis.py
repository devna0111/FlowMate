import cv2
import numpy as np
import os

def analyze_visual_features(video_path: str, frame_skip: int = 5, resize_dim=(640, 360)) -> dict:
    cap = cv2.VideoCapture(video_path)

    # OpenCV 얼굴 검출기 초기화 (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 몸체 검출기 (상체 검출용)
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

    frame_count = 0
    analyzed_frames = 0
    face_detected = 0
    gesture_detected = 0
    
    # 움직임 감지를 위한 이전 프레임 저장
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # frame_skip 단위로 샘플링
        if frame_count % frame_skip != 0:
            continue

        # 해상도 축소
        frame_resized = cv2.resize(frame, resize_dim)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            face_detected += 1

        # 상체/움직임 검출
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        
        # 추가로 움직임 감지 (제스처 대용)
        motion_detected = False
        if prev_gray is not None:
            # 프레임 차이를 이용한 움직임 감지
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_pixels = cv2.countNonZero(thresh)
            
            # 프레임 크기의 5% 이상 변화가 있으면 움직임으로 판단
            if motion_pixels > (resize_dim[0] * resize_dim[1] * 0.05):
                motion_detected = True
        
        # 몸체 검출 또는 유의미한 움직임이 있으면 제스처로 판단
        if len(bodies) > 0 or motion_detected:
            gesture_detected += 1

        prev_gray = gray.copy()
        analyzed_frames += 1

    cap.release()

    return {
        "total_frames": analyzed_frames,
        "face_detected_frames": face_detected,
        "gesture_detected_frames": gesture_detected,
        "face_detection_ratio": round(face_detected / analyzed_frames, 2) if analyzed_frames else 0.0,
        "gesture_ratio": round(gesture_detected / analyzed_frames, 2) if analyzed_frames else 0.0,
    }

# ✅ 테스트 실행
if __name__ == "__main__":
    import sys
    import os

    default_video = "uploads/sample_video.mp4"
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video

    if not os.path.exists(video_path):
        print(f"영상 파일이 존재하지 않습니다: {video_path}")
    else:
        print(f"분석 중인 영상 파일: {video_path}\n")
        result = analyze_visual_features(video_path)

        print("🎥 시각 표현 분석 결과:")
        for k, v in result.items():
            print(f"- {k}: {v}")
