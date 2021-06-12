# DhivehiSTT
This repository contains code to run STT model used in the [DV-Subs](https://github.com/DhivehiAI/DV-Subs) project by [Dhivehi.ai](https://dhivehi.ai/)

## Setup
1. ```git clone https://github.com/ashraq1455/DhivehiSTT.git```
2. ```pip install -r requirements.txt```
3. Download the model from [here](https://dhivehi.ai/docs/technologies/stt/)
4. Extract and copy ```wav2vec_traced_quantized.pt``` and ```vocab.json``` to ```models``` directory

## How to use
```python
from stt import transcribe
transcription = transcribe(audio_bytes)
```
