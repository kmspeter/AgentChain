import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import config
from utils.data_loader import load_dataset_from_files
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

class FinetuneService:
    def __init__(self):
        """파인튜닝 서비스 초기화"""
        self.finetune_status = {}  # 모델별 파인튜닝 상태 저장
    
    def get_status(self, model_name):
        """모델 파인튜닝 상태 반환"""
        if model_name not in self.finetune_status:
            return {"status": "not_started", "progress": 0}
        return self.finetune_status[model_name]
    
    def _set_status(self, model_name, status, progress=0, message=""):
        """모델 파인튜닝 상태 설정"""
        self.finetune_status[model_name] = {
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": time.time()
        }
    
    def finetune_model(self, model_name, new_model_name, file_paths, epochs=None):
        """모델 파인튜닝 실행"""
        if epochs is None:
            epochs = config.NUM_EPOCHS
            
        try:
            # 새 모델 경로 설정
            output_dir = os.path.join(config.MODELS_DIR, new_model_name, "model_files")
            os.makedirs(output_dir, exist_ok=True)
            
            # 상태 업데이트
            self._set_status(new_model_name, "loading_model", 5, "모델 로딩 중...")
            
            # 모델 로드
            if os.path.exists(os.path.join(config.MODELS_DIR, model_name)):
                base_model_path = os.path.join(config.MODELS_DIR, model_name)
            else:
                base_model_path = model_name  # HuggingFace 모델 ID로 가정
            
            # 모델과 토크나이저 로드
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto" if torch.cuda.is_available() else None
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            
            # 패딩 토큰 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # 상태 업데이트
            self._set_status(new_model_name, "processing_data", 15, "데이터 처리 중...")
            
            # 데이터 로드 및 전처리
            texts = load_dataset_from_files(file_paths)
            
            # 토크나이징 함수 정의
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)
            
            # 데이터셋 클래스 정의
            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer):
                    self.encodings = [tokenizer(text, return_tensors="pt", padding="max_length", 
                                              truncation=True, max_length=512) for text in texts]
                
                def __getitem__(self, idx):
                    item = {key: val[0] for key, val in self.encodings[idx].items()}
                    item["labels"] = item["input_ids"].clone()
                    return item
                
                def __len__(self):
                    return len(self.encodings)
            
            # 데이터셋 생성
            dataset = TextDataset(texts, tokenizer)
            
            # 상태 업데이트
            self._set_status(new_model_name, "configuring_lora", 25, "LoRA 설정 중...")
            
            # LoRA 설정
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=["q_proj", "v_proj"]  # 모델에 따라 조정 필요
            )
            
            # LoRA 모델 준비
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            
            # 데이터 콜레이터 설정
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # 상태 업데이트
            self._set_status(new_model_name, "training", 30, "학습 시작...")
            
            # 학습 인수 설정
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=config.BATCH_SIZE,
                gradient_accumulation_steps=1,
                learning_rate=config.LEARNING_RATE,
                weight_decay=0.01,
                save_strategy="epoch",
                save_total_limit=1,
                logging_steps=10,
                logging_dir=os.path.join(output_dir, "logs"),
                report_to="none"
            )
            
            # 학습 콜백 정의
            class StatusCallback(TrainerCallback):
                def __init__(self, service, model_name, epochs):
                    self.service = service
                    self.model_name = model_name
                    self.epochs = epochs

                def on_init_end(self, args, state, control, **kwargs):
                    self.service._set_status(
                        self.model_name,
                        "initialized",
                        30,
                        "Trainer 초기화 완료"
                    )

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if state.epoch is not None:
                        progress = min(95, 30 + (state.epoch / self.epochs) * 65)
                        self.service._set_status(
                            self.model_name,
                            "training",
                            progress,
                            f"에폭 {state.epoch:.2f}/{self.epochs} 학습 중..."
                        )
            
            # 트레이너 설정
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[StatusCallback(self, new_model_name, epochs)]
            )
            
            # 학습 실행
            trainer.train()
            
            # 모델 저장
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # 어댑터 구성 파일에 기본 모델 경로 추가
            adapter_config_path = os.path.join(output_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                
                adapter_config["base_model_name_or_path"] = base_model_path
                
                with open(adapter_config_path, "w") as f:
                    json.dump(adapter_config, f, indent=2)
            
            # 상태 업데이트
            self._set_status(new_model_name, "completed", 100, "파인튜닝 완료")
            
            return {"status": "success", "message": f"모델 {new_model_name} 파인튜닝 완료"}
            
        except Exception as e:
            # 오류 상태 업데이트
            error_message = str(e)
            self._set_status(new_model_name, "failed", 0, f"오류: {error_message}")
            
            return {"status": "error", "message": error_message}