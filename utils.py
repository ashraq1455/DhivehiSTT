import os
import torch
import librosa
from scipy.io import wavfile
from pyAudioAnalysis.audioBasicIO import read_audio_file
from pyAudioAnalysis.audioSegmentation import silence_removal
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


class STTPipeline:
    def __init__(self, m_path):
        self.stt_model_path = os.path.join(m_path, "wav2vec_traced_quantized.pt")
        self.stt_vocab_file = os.path.join(m_path, "vocab.json")
        self.sampling_rate = 16000

        print("Initializing STT Model")
        tokenizer = Wav2Vec2CTCTokenizer(self.stt_vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.sampling_rate, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.model = torch.jit.load(self.stt_model_path)

    def __call__(self, audio_path):
        audio_input, sr = librosa.load(audio_path, sr=self.sampling_rate)
        inputs = self.processor(
            audio_input,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values)['logits']

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


def extractAudio(input_file, output_dir, smoothing_window = 1.0, weight = 0.1):
    print("Detecting silences...")
    [fs, x] = read_audio_file(input_file)
    segmentLimits = silence_removal(x, fs, 0.05, 0.05, smoothing_window, weight)
    ifile_name = os.path.basename(input_file)

    os.makedirs(output_dir, exist_ok=True)
    files = []

    print("Writing segments...")
    for s in segmentLimits:
        strOut = "{0:s}_{1:.3f}-{2:.3f}.wav".format(ifile_name, s[0], s[1])
        strOut = os.path.join(output_dir, strOut)
        wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])
        files.append(strOut)

    return files