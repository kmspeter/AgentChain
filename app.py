from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import uvicorn
import shutil
from pydantic import BaseModel

from services.model_service import ModelService
from services.finetune_service import FinetuneService
from services.rag_service import RagService
from utils.data_loader import save_upload_file
import config

app = FastAPI(title="모델 서빙 및 파인튜닝 API")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 인스턴스 생성
model_service = ModelService()
finetune_service = FinetuneService()
rag_service = RagService()

# 시스템 초기화
config.init_system()

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.get("/")
def read_root():
    return {"message": "모델 서빙 및 파인튜닝 API가 실행 중입니다"}

@app.get("/models")
def list_models():
    """사용 가능한 모델 목록 반환"""
    return model_service.list_models()

@app.post("/generate")
async def generate_text(model_name: str, request: TextGenerationRequest):
    """텍스트 생성"""
    try:
        result = model_service.generate(model_name, request.prompt, request.max_length, request.temperature)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finetune")
async def finetune_model(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    new_model_name: str = Form(...),
    files: List[UploadFile] = File(...),
    epochs: int = Form(config.NUM_EPOCHS)
):
    """모델 파인튜닝"""
    try:
        # 파일 저장
        file_paths = []
        for file in files:
            if not any(file.filename.endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
                raise HTTPException(status_code=400, detail=f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(config.ALLOWED_EXTENSIONS)}")
            
            temp_file_path = save_upload_file(file)
            file_paths.append(temp_file_path)
        
        # 백그라운드에서 파인튜닝 작업 실행
        background_tasks.add_task(
            finetune_service.finetune_model,
            model_name=model_name,
            new_model_name=new_model_name,
            file_paths=file_paths,
            epochs=epochs
        )
        
        return {"message": f"모델 {model_name}의 파인튜닝이 시작되었습니다. 새 모델명: {new_model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-rag")
async def apply_rag(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    new_model_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """RAG 적용"""
    try:
        # 파일 저장
        file_paths = []
        for file in files:
            if not any(file.filename.endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
                raise HTTPException(status_code=400, detail=f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(config.ALLOWED_EXTENSIONS)}")
            
            temp_file_path = save_upload_file(file)
            file_paths.append(temp_file_path)
        
        # 백그라운드에서 RAG 적용 작업 실행
        background_tasks.add_task(
            rag_service.apply_rag,
            model_name=model_name,
            new_model_name=f"{new_model_name}_rag",
            file_paths=file_paths
        )
        
        return {"message": f"모델 {model_name}에 RAG 적용이 시작되었습니다. 새 모델명: {new_model_name}_rag"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/finetune-status/{model_name}")
async def finetune_status(model_name: str):
    """파인튜닝 상태 확인"""
    return finetune_service.get_status(model_name)

@app.get("/rag-status/{model_name}")
async def rag_status(model_name: str):
    """RAG 적용 상태 확인"""
    return rag_service.get_status(model_name)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)