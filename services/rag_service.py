import os
import shutil
import time

# 반드시 최상단에서 import
import sentence_transformers

import sqlite3
import sys
sys.modules['sqlite3'] = sqlite3
sys.modules['sqlite3.dbapi2'] = sqlite3

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from utils.data_loader import load_dataset_from_files
import config

class RagService:
    def __init__(self):
        """RAG 서비스 초기화"""
        self.rag_status = {}  # 모델별 RAG 적용 상태 저장
    
    def get_status(self, model_name):
        """모델 RAG 적용 상태 반환"""
        if model_name not in self.rag_status:
            return {"status": "not_started", "progress": 0}
        return self.rag_status[model_name]
    
    def _set_status(self, model_name, status, progress=0, message=""):
        """모델 RAG 적용 상태 설정"""
        self.rag_status[model_name] = {
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": time.time()
        }
    
    def apply_rag(self, model_name, new_model_name, file_paths):
        """RAG 적용 실행"""
        try:
            # 새 모델 디렉토리 설정
            model_dir = os.path.join(config.MODELS_DIR, new_model_name)
            vector_store_dir = os.path.join(model_dir, "vector_store")
            
            # 디렉토리 생성
            os.makedirs(model_dir, exist_ok=True)
            
            # 상태 업데이트
            self._set_status(new_model_name, "loading_data", 10, "데이터 로딩 중...")
            
            # 데이터 로드
            texts = load_dataset_from_files(file_paths)
            
            # 상태 업데이트
            self._set_status(new_model_name, "splitting_text", 30, "텍스트 분할 중...")
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len
            )
            
            # 텍스트를 문서 형식으로 변환
            from langchain.schema import Document
            documents = []
            for i, text in enumerate(texts):
                doc = Document(
                    page_content=text,
                    metadata={"source": f"document_{i}"}
                )
                documents.append(doc)
            
            # 문서 분할
            splits = text_splitter.split_documents(documents)
            
            # 상태 업데이트
            self._set_status(new_model_name, "creating_embeddings", 50, "임베딩 생성 중...")
            
            # 임베딩 모델 로드
            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            
            # 기존 벡터 스토어 제거 (존재하는 경우)
            if os.path.exists(vector_store_dir):
                shutil.rmtree(vector_store_dir)
            
            # 벡터 스토어 생성
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=vector_store_dir
            )
            
            # 벡터 스토어 영구 저장
            vector_store.persist()
            
            # 원본 모델 링크 생성 (원본 모델이 로컬에 있는 경우)
            original_model_path = os.path.join(config.MODELS_DIR, model_name)
            if os.path.exists(original_model_path):
                model_files_dir = os.path.join(original_model_path, "model_files")
                if os.path.exists(model_files_dir):
                    target_model_files_dir = os.path.join(model_dir, "model_files")
                    if not os.path.exists(target_model_files_dir):
                        shutil.copytree(model_files_dir, target_model_files_dir)
            
            # 상태 업데이트
            self._set_status(new_model_name, "completed", 100, "RAG 적용 완료")
            
            return {"status": "success", "message": f"모델 {model_name}에 RAG 적용 완료, 새 모델: {new_model_name}"}
            
        except Exception as e:
            # 오류 상태 업데이트
            error_message = str(e)
            self._set_status(new_model_name, "failed", 0, f"오류: {error_message}")
            return {"status": "error", "message": error_message}
