import shutil
from tqdm import tqdm
from utils import STTPipeline, extractAudio


input = "clip.mp3"
temp_dir = "temp"
model_dir = "model"
silence_window = 1.0
silence_weight = 0.3

stt = STTPipeline(model_dir)

files = extractAudio(
        input,
        temp_dir,
        smoothing_window=silence_window,
        weight=silence_weight
    )

completed_text = []
for w_file in tqdm(files):
    transcription = stt(w_file)
    if len(transcription.strip()) == 0:
        continue

    completed_text.append(transcription)

print(completed_text)

print("Removing temporary files...")
shutil.rmtree(temp_dir)