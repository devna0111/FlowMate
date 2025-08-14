# import torch
# import faster_whisper

# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"GPU count: {torch.cuda.device_count()}")
# if torch.cuda.is_available():
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")

# # faster-whisper 테스트
# from faster_whisper import WhisperModel
# model = WhisperModel("base", device="cuda", compute_type="float16")
# print("faster-whisper GPU 모델 로드 성공!")

# import tensorflow as tf
# print("현재 GPU 감지:", tf.config.list_physical_devices('GPU'))

import sklearn as skl

print(skl.__version__)
