import os
import tempfile
import json
import json5
from typing import List, Dict, Any
from fastapi import UploadFile
import config

# PDF 로더
def load_pdf(file_path):
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF 로드 실패: {str(e)}")
        return ""

# TXT 로더
def load_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # utf-8 로딩 실패 시 다른 인코딩 시도
        try:
            with open(file_path, 'r', encoding='cp949') as file:
                return file.read()
        except Exception as e:
            print(f"텍스트 파일 로드 실패: {str(e)}")
            return ""
    except Exception as e:
        print(f"텍스트 파일 로드 실패: {str(e)}")
        return ""

# JSON 로더
def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except:
                file.seek(0)
                data = json5.load(file)

        texts = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # instruction + output 결합
                    if "instruction" in item and "output" in item:
                        prompt = f"### 질문: {item['instruction']}\n### 답변: {item['output']}"
                        texts.append(prompt)
                    elif "text" in item:
                        texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            return texts

        elif isinstance(data, dict):
            if "text" in data:
                return [data["text"]]
            elif "content" in data:
                return [data["content"]]
            else:
                return [json.dumps(data, ensure_ascii=False, indent=2)]

        return []

    except Exception as e:
        print(f"JSON 파일 로드 실패: {str(e)}")
        return []

# 파일 형식에 따른 로더 매핑
FILE_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".json": load_json,
}

def save_upload_file(upload_file: UploadFile) -> str:
    """업로드된 파일을 임시 디렉토리에 저장"""
    try:
        # 임시 파일 생성
        suffix = os.path.splitext(upload_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # 업로드된 파일 내용 복사
            content = upload_file.file.read()
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        raise Exception(f"파일 저장 실패: {str(e)}")
    finally:
        upload_file.file.close()

def load_dataset_from_files(file_paths: List[str]) -> List[str]:
    """여러 파일에서 데이터셋 로드"""
    texts = []
    
    for file_path in file_paths:
        # 파일 확장자 확인
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 지원하는 파일 형식인지 확인
        if file_ext not in FILE_LOADERS:
            print(f"지원하지 않는 파일 형식: {file_ext}")
            continue
        
        # 적절한 로더로 텍스트 추출
        loader = FILE_LOADERS[file_ext]
        text = loader(file_path)
        
        if text:
            texts.append(text)
    
    return texts