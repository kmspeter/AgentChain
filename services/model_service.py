import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import config
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class ModelService:
    def __init__(self):
        """모델 서비스 초기화"""
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.rag_chains = {}
        
    def list_models(self):
        """사용 가능한 모델 목록 반환"""
        try:
            models = []
            if os.path.exists(config.MODELS_DIR):
                for model_name in os.listdir(config.MODELS_DIR):
                    model_dir = os.path.join(config.MODELS_DIR, model_name)
                    if os.path.isdir(model_dir):
                        models.append({
                            "name": model_name,
                            "has_rag": os.path.exists(os.path.join(model_dir, "vector_store"))
                        })
            return {"models": models}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_model_path(self, model_name):
        local_path = os.path.join(config.MODELS_DIR, model_name)
        return local_path if os.path.exists(local_path) else model_name
    
    def load_model(self, model_name):
        """모델 및 토크나이저 로드"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        
        model_path = self._get_model_path(model_name)
        adapter_path = os.path.join(model_path, "model_files", "adapter_config.json")
        base_model_path = model_path
        
        # LoRA 체크
        if os.path.exists(adapter_path):
            with open(adapter_path, "r") as f:
                cfg = json.load(f)
                base_model_path = cfg.get("base_model_name_or_path", model_path)
            # device_map 제거
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(
                base_model,
                os.path.join(model_path, "model_files")
            )
        else:
            # device_map 제거
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # CUDA 사용 시 모델을 GPU로 이동
        if torch.cuda.is_available():
            model.to("cuda")
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.loaded_models[model_name] = model
        self.loaded_tokenizers[model_name] = tokenizer
        return model, tokenizer
    
    def generate(self, model_name, prompt, max_length=100, temperature=0.7):
        """텍스트 생성 (pipeline 대신 직접 generate 호출)"""
        try:
            # RAG 모델이면 별도 처리
            if model_name.endswith("_rag") and model_name in self.rag_chains:
                return self._generate_with_rag(model_name, prompt)
            
            model, tokenizer = self.load_model(model_name)
            model.eval()
            with torch.no_grad():
                # 토크나이저에서 truncation + padding
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # max_new_tokens로 오직 새로 생성할 토큰 수만 지정
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트 부분 제거
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text
        
        except Exception as e:
            raise Exception(f"텍스트 생성 실패: {e}")
    
    def load_rag_chain(self, model_name):
        """RAG 체인 로드"""
        if model_name in self.rag_chains:
            return self.rag_chains[model_name]
        
        base_name = model_name.replace("_rag", "")
        model_path = self._get_model_path(base_name)
        vector_store_path = os.path.join(config.MODELS_DIR, model_name, "vector_store")
        if not os.path.exists(vector_store_path):
            raise Exception(f"벡터 스토어를 찾을 수 없습니다: {vector_store_path}")
        
        model, tokenizer = self.load_model(base_name)
        
        # RAG용 pipeline: device=0으로 GPU 지정, CPU는 -1
        device = 0 if torch.cuda.is_available() else -1
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        self.rag_chains[model_name] = qa_chain
        return qa_chain
    
    def _generate_with_rag(self, model_name, query):
        try:
            qa_chain = self.load_rag_chain(model_name)
            result = qa_chain({"query": query})
            return result["result"]
        except Exception as e:
            raise Exception(f"RAG 생성 실패: {e}")
