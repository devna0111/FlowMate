import cv2
import numpy as np
import os

# MediaPipe import with fallback to OpenCV
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe를 사용합니다.")
except ImportError as e:
    print(f"MediaPipe를 가져올 수 없습니다: {e}")
    print("OpenCV fallback을 사용합니다.")
    MEDIAPIPE_AVAILABLE = False

def analyze_visual_features(video_path: str, frame_skip: int = 5, resize_dim=(640, 360)) -> dict:
    cap = cv2.VideoCapture(video_path)

    if MEDIAPIPE_AVAILABLE:
        return _analyze_with_mediapipe(cap, frame_skip, resize_dim)
    else:
        return _analyze_with_opencv(cap, frame_skip, resize_dim)

def _analyze_with_mediapipe(cap, frame_skip: int, resize_dim: tuple) -> dict:
    """MediaPipe를 사용한 분석"""
    # MediaPipe 초기화
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    # MediaPipe 모델 초기화
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    frame_count = 0
    analyzed_frames = 0
    face_detected = 0
    pose_detected = 0
    hand_gesture_detected = 0
    
    # 제스처 분석을 위한 변수들
    previous_landmarks = None
    movement_threshold = 0.05

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_resized = cv2.resize(frame, resize_dim)
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 얼굴 검출
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            face_detected += 1

        # 포즈 검출
        pose_results = pose.process(rgb_frame)
        current_pose_detected = False
        if pose_results.pose_landmarks:
            pose_detected += 1
            current_pose_detected = True
            
            # 포즈 랜드마크를 이용한 움직임 분석
            current_landmarks = []
            for landmark in pose_results.pose_landmarks.landmark:
                current_landmarks.append([landmark.x, landmark.y])
            
            if previous_landmarks is not None:
                movement = np.array(current_landmarks) - np.array(previous_landmarks)
                avg_movement = np.mean(np.abs(movement))
                if avg_movement > movement_threshold:
                    hand_gesture_detected += 1
            
            previous_landmarks = current_landmarks

        # 손 제스처 검출
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            if not current_pose_detected:
                hand_gesture_detected += 1

        analyzed_frames += 1

    cap.release()
    face_detection.close()
    pose.close() 
    hands.close()

    return {
        "total_frames": analyzed_frames,
        "face_detected_frames": face_detected,
        "pose_detected_frames": pose_detected,
        "gesture_detected_frames": hand_gesture_detected,
        "face_detection_ratio": round(face_detected / analyzed_frames, 2) if analyzed_frames else 0.0,
        "pose_detection_ratio": round(pose_detected / analyzed_frames, 2) if analyzed_frames else 0.0,
        "gesture_ratio": round(hand_gesture_detected / analyzed_frames, 2) if analyzed_frames else 0.0,
    }

def _analyze_with_opencv(cap, frame_skip: int, resize_dim: tuple) -> dict:
    """OpenCV를 사용한 fallback 분석"""
    # OpenCV 분류기 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

    frame_count = 0
    analyzed_frames = 0
    face_detected = 0
    gesture_detected = 0
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_resized = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            face_detected += 1

        # 상체/움직임 검출
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        
        # 움직임 감지
        motion_detected = False
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_pixels = cv2.countNonZero(thresh)
            
            if motion_pixels > (resize_dim[0] * resize_dim[1] * 0.05):
                motion_detected = True
        
        if len(bodies) > 0 or motion_detected:
            gesture_detected += 1

        prev_gray = gray.copy()
        analyzed_frames += 1

    cap.release()

    return {
        "total_frames": analyzed_frames,
        "face_detected_frames": face_detected,
        "pose_detected_frames": 0,  # OpenCV에서는 포즈 검출 불가
        "gesture_detected_frames": gesture_detected,
        "face_detection_ratio": round(face_detected / analyzed_frames, 2) if analyzed_frames else 0.0,
        "pose_detection_ratio": 0.0,  # OpenCV에서는 포즈 검출 불가
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
