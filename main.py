import uvicorn
from stt import transcribe
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/predict/")
async def inference(audio_bytes: bytes = File(...)):
    text = transcribe(audio_bytes)
    print(text)
    return JSONResponse({"text": text})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)