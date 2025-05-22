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
            # 모델 디렉토리 탐색
            models = []
            if os.path.exists(config.MODELS_DIR):
                for model_name in os.listdir(config.MODELS_DIR):
                    model_dir = os.path.join(config.MODELS_DIR, model_name)
                    if os.path.isdir(model_dir):
                        model_info = {
                            "name": model_name,
                            "has_rag": os.path.exists(os.path.join(model_dir, "vector_store"))
                        }
                        models.append(model_info)
            
            return {"models": models}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_model_path(self, model_name):
        """모델 경로 반환"""
        # 로컬 저장소에 있는지 확인
        local_path = os.path.join(config.MODELS_DIR, model_name)
        if os.path.exists(local_path):
            return local_path
        
        # 아니면 HuggingFace 모델로 간주
        return model_name
    
    def load_model(self, model_name):
        """모델 및 토크나이저 로드"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        
        try:
            model_path = self._get_model_path(model_name)
            
            # LoRA 파인튜닝 모델인지 확인
            adapter_path = os.path.join(model_path, "model_files", "adapter_config.json")
            base_model_path = model_path
            
            if os.path.exists(adapter_path):
                # adapter.json 파일에서 기본 모델 이름 가져오기
                with open(os.path.join(model_path, "model_files", "adapter_config.json"), "r") as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path", model_path)
                
                # 기본 모델 로드
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path, 
                    device_map="auto" if torch.cuda.is_available() else None
                )
                # LoRA 어댑터 로드
                model = PeftModel.from_pretrained(
                    base_model,
                    os.path.join(model_path, "model_files")
                )
            else:
                # 일반 모델 로드
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.loaded_models[model_name] = model
            self.loaded_tokenizers[model_name] = tokenizer
            
            return model, tokenizer
        
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")
    
    def generate(self, model_name, prompt, max_length=100, temperature=0.7):
        """텍스트 생성"""
        try:
            # RAG 모델인지 확인
            if model_name.endswith("_rag") and model_name in self.rag_chains:
                return self._generate_with_rag(model_name, prompt)
            
            # 일반 생성
            model, tokenizer = self.load_model(model_name)
            
            # 텍스트 생성 파이프라인 설정
            text_gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            
            # 텍스트 생성
            result = text_gen(prompt)[0]["generated_text"]
            
            # 프롬프트를 제외하고 생성된 텍스트만 반환 (선택적)
            if result.startswith(prompt):
                result = result[len(prompt):]
                
            return result
            
        except Exception as e:
            raise Exception(f"텍스트 생성 실패: {str(e)}")
    
    def load_rag_chain(self, model_name):
        """RAG 체인 로드"""
        if model_name in self.rag_chains:
            return self.rag_chains[model_name]
        
        try:
            # RAG 모델 이름에서 _rag 제거
            base_name = model_name.replace("_rag", "")
            model_path = self._get_model_path(base_name)
            vector_store_path = os.path.join(config.MODELS_DIR, model_name, "vector_store")
            
            # 벡터 스토어 존재 확인
            if not os.path.exists(vector_store_path):
                raise Exception(f"벡터 스토어를 찾을 수 없습니다: {vector_store_path}")
            
            # 모델 로드
            model, tokenizer = self.load_model(base_name)
            
            # HuggingFace 파이프라인 생성
            llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            
            llm = HuggingFacePipeline(pipeline=llm_pipeline)
            
            # 임베딩 모델 로드
            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            
            # 벡터 스토어 로드
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings
            )
            
            # RAG 체인 생성
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=True
            )
            
            self.rag_chains[model_name] = qa_chain
            return qa_chain
            
        except Exception as e:
            raise Exception(f"RAG 체인 로드 실패: {str(e)}")
    
    def _generate_with_rag(self, model_name, query):
        """RAG를 사용한 텍스트 생성"""
        try:
            qa_chain = self.load_rag_chain(model_name)
            result = qa_chain({"query": query})
            return result["result"]
        except Exception as e:
            raise Exception(f"RAG 생성 실패: {str(e)}")