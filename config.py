import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 모델 관련 설정
DEFAULT_MODEL = "llama3.2_1b"  # 기본 모델 지정 (사용자 환경에 맞게 변경)

# 파인튜닝 관련 설정
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3

# RAG 관련 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 기본 임베딩 모델

# 허용되는 파일 형식
ALLOWED_EXTENSIONS = [".pdf", ".txt", ".json"]

# 시스템 초기화 함수
def init_system():
    """시스템 초기화 - 필요한 디렉토리 생성"""
    os.makedirs(MODELS_DIR, exist_ok=True)