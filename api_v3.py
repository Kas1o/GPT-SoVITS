import os
import sys
import traceback
import torch
import numpy as np
import yaml
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from io import BytesIO
import soundfile as sf
from inference_webui import get_tts_wav, change_sovits_weights, change_gpt_weights


CONFIG_PATH = "GPT_SoVITS/configs/tts_infer.yaml"

# 读取配置文件
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# 设置默认权重路径
gpt_weights_path = config["custom"]["t2s_weights_path"]
sovits_weights_path = config["custom"]["vits_weights_path"]
change_gpt_weights(gpt_weights_path)
change_sovits_weights(sovits_weights_path)

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_lang: str
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut0"
    batch_size: int = 1
    batch_threshold: float = 0.75
    speed_factor: float = 1.0
    streaming_mode: bool = False
    media_type: str = "wav"

@app.get("/tts")
async def tts_get_endpoint(
        text: str,
        text_lang: str,
        ref_audio_path: str,
        prompt_lang: str,
        prompt_text: str = "",
        top_k: int = 5,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut0",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        speed_factor: float = 1.0,
        streaming_mode: bool = False,
        media_type: str = "wav"
):
    try:
        sr, audio_data = next(get_tts_wav(ref_audio_path, prompt_text, prompt_lang, text, text_lang, text_split_method, top_k, top_p, temperature))
        audio_bytes = BytesIO()
        sf.write(audio_bytes, audio_data, sr, format=media_type)
        audio_bytes.seek(0)
        return StreamingResponse(audio_bytes, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "TTS failed", "Exception": str(e)})

@app.post("/tts")
async def tts_post_endpoint(request: TTSRequest):
    try:
        req = request.dict()
        sr, audio_data = next(get_tts_wav(req["ref_audio_path"], req["prompt_text"], req["prompt_lang"], req["text"], req["text_lang"], req["text_split_method"], req["top_k"], req["top_p"], req["temperature"], parse_language=False))
        audio_bytes = BytesIO()
        sf.write(audio_bytes, audio_data, sr, format=req['media_type'])
        audio_bytes.seek(0)
        return StreamingResponse(audio_bytes, media_type=f"audio/{req['media_type']}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "TTS failed", "Exception": str(e)})

@app.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str):
    try:
        change_gpt_weights(weights_path)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

@app.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str):
    try:
        change_sovits_weights(weights_path)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9880)
