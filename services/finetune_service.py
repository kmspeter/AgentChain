import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import config
from utils.data_loader import load_dataset_from_files

class FinetuneService:
    def __init__(self):
        """파인튜닝 서비스 초기화"""
        self.finetune_status = {}  # 모델별 파인튜닝 상태 저장
    
    def get_status(self, model_name):
        """모델 파인튜닝 상태 반환"""
        return self.finetune_status.get(model_name, {"status": "not_started", "progress": 0})
    
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
            # 1) 출력 디렉토리 변경
            output_dir = os.path.join(config.MODELS_DIR, new_model_name)
            os.makedirs(output_dir, exist_ok=True)
            self._set_status(new_model_name, "loading_model", 5, "모델 로딩 중...")

            # 2) 모델 & 토크나이저 로드 (device_map 제거)
            if os.path.exists(os.path.join(config.MODELS_DIR, model_name)):
                base_model_path = os.path.join(config.MODELS_DIR, model_name)
            else:
                base_model_path = model_name

            model = AutoModelForCausalLM.from_pretrained(base_model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if torch.cuda.is_available():
                model.to("cuda")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 3) 데이터 로드 및 전처리
            self._set_status(new_model_name, "processing_data", 15, "데이터 처리 중...")
            texts = load_dataset_from_files(file_paths)

            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer):
                    self.encodings = [
                        tokenizer(
                            text,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=512
                        )
                        for text in texts
                    ]
                def __getitem__(self, idx):
                    item = {k: v[0] for k, v in self.encodings[idx].items()}
                    item["labels"] = item["input_ids"].clone()
                    return item
                def __len__(self):
                    return len(self.encodings)

            dataset = TextDataset(texts, tokenizer)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # 4) LoRA 설정
            self._set_status(new_model_name, "configuring_lora", 25, "LoRA 설정 중...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=["q_proj", "v_proj"]
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            # 5) 옵티마이저 & 스케줄러 & 데이터로더
            self._set_status(new_model_name, "training", 30, "학습 시작...")
            dataloader = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=data_collator
            )
            optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
            total_steps = epochs * len(dataloader)
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

            # 6) 수동 학습 루프 (Accelerate 불필요)
            global_step = 0
            for epoch in range(1, epochs + 1):
                model.train()
                for batch in dataloader:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                # 에폭 단위 상태 업데이트
                progress = int(30 + (epoch / epochs) * 65)
                self._set_status(
                    new_model_name,
                    "training",
                    progress,
                    f"에폭 {epoch}/{epochs} 완료"
                )

            # 7) 모델 저장
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # 8) adapter_config.json 업데이트
            adapter_config_path = os.path.join(output_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_cfg = json.load(f)
                adapter_cfg["base_model_name_or_path"] = base_model_path
                with open(adapter_config_path, "w") as f:
                    json.dump(adapter_cfg, f, indent=2)

            self._set_status(new_model_name, "completed", 100, "파인튜닝 완료")
            return {"status": "success", "message": f"모델 {new_model_name} 파인튜닝 완료"}

        except Exception as e:
            err = str(e)
            self._set_status(new_model_name, "failed", 0, f"오류: {err}")
            return {"status": "error", "message": err}
