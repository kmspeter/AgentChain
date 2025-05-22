from pyngrok import ngrok
import uvicorn
import time
import os

# reloader가 중복 실행되는 걸 방지
if __name__ == "__main__" and os.getenv("RUN_MAIN") != "true":
    # FastAPI 서버 포트
    port = 8000

    # ngrok 터널 생성
    public_url = ngrok.connect(port)
    print(f" * Ngrok public URL: {public_url}")

    # 1초 대기 (안정성 보장용)
    time.sleep(1)

    # FastAPI 서버 실행 (reload=True는 유지 가능)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
