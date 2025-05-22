import os
import json
import torch
from typing import Dict, Any, List, Optional
import config

def check_model_exists(model_name: str) -> bool:
    """모델이 로컬 저장소에 존재하는지 확인"""
    model_path = os.path.join(config.MODELS_DIR, model_name)
    return os.path.exists(model_path)

def get_model_info(model_name: str) -> Dict[str, Any]:
    """모델 정보 반환"""
    model_path = os.path.join(config.MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        return {"exists": False}
    
    # 기본 정보
    info = {
        "exists": True,
        "name": model_name,
        "has_rag": os.path.exists(os.path.join(model_path, "vector_store")),
        "is_finetuned": False,
        "base_model": None
    }
    
    # 파인튜닝 정보 확인
    adapter_config_path = os.path.join(model_path, "model_files", "adapter_config.json")
    if os.path.exists(adapter_config_path):
        info["is_finetuned"] = True
        try:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                if "base_model_name_or_path" in adapter_config:
                    info["base_model"] = adapter_config["base_model_name_or_path"]
        except:
            pass
    
    return info

def list_available_models() -> List[Dict[str, Any]]:
    """사용 가능한 모델 목록 반환"""
    models = []
    
    # 로컬 모델 탐색
    if os.path.exists(config.MODELS_DIR):
        for model_name in os.listdir(config.MODELS_DIR):
            model_path = os.path.join(config.MODELS_DIR, model_name)
            if os.path.isdir(model_path):
                info = get_model_info(model_name)
                models.append(info)
    
    return models

def device_info() -> Dict[str, Any]:
    """시스템 장치 정보 반환"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    # GPU 정보
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(info["current_device"])
        info["memory_allocated"] = torch.cuda.memory_allocated(info["current_device"]) / (1024 ** 3)  # GB
        info["memory_reserved"] = torch.cuda.memory_reserved(info["current_device"]) / (1024 ** 3)  # GB
    
    return info

def get_compatible_target_modules(model_name: str) -> List[str]:
    """모델에 적합한 LoRA 타겟 모듈 목록 반환"""
    # 일반적인 타겟 모듈 맵핑
    target_modules_map = {
        "gpt2": ["c_attn", "c_proj"],
        "opt": ["q_proj", "v_proj"],
        "llama": ["q_proj", "v_proj"],
        "bloom": ["query_key_value"],
        "default": ["query", "value"]
    }
    
    # 모델 아키텍처 유형 감지
    for arch, modules in target_modules_map.items():
        if arch in model_name.lower():
            return modules
    
    return target_modules_map["default"]