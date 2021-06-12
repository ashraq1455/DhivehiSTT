import shutil
from utils import STTPipeline, extractAudio


model_dir = "model"
silence_window = 1.0
silence_weight = 0.3

stt = STTPipeline(model_dir)


def transcribe(audio_bytes):
    files = extractAudio(
            audio_bytes,
            smoothing_window=silence_window,
            weight=silence_weight
        )

    completed_text = []
    for w_file in files:
        transcription = stt(w_file)
        if len(transcription.strip()) == 0:
            continue

        completed_text.append(transcription)

    shutil.rmtree("temp")
    return "".join(completed_text)