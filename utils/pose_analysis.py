import cv2
import mediapipe as mp

def analyze_visual_features(video_path: str, frame_skip: int = 5, resize_dim=(640, 360)) -> dict:
    cap = cv2.VideoCapture(video_path)

    # GPU 최적화 옵션 활성화 및 초기화 옵션 최적화
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    frame_count = 0
    analyzed_frames = 0
    face_detected = 0
    gesture_detected = 0

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

        # RGB 변환
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Mediapipe 분석 수행
        face_result = mp_face.process(frame_rgb)
        pose_result = mp_pose.process(frame_rgb)
        hands_result = mp_hands.process(frame_rgb)

        # 얼굴이 인식된 프레임 수
        if face_result.multi_face_landmarks:
            face_detected += 1

        # 자세 또는 손동작이 감지된 프레임 수
        if pose_result.pose_landmarks or hands_result.multi_hand_landmarks:
            gesture_detected += 1

        analyzed_frames += 1

    cap.release()
    mp_face.close()
    mp_pose.close()
    mp_hands.close()

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
