from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cpu", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")
audio_path = r'/Users/chrvt/Downloads/Welcome-and-4-Qualifiers_1.mp3'
segments, info = model.transcribe(audio_path, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
segments = list(segments)

import pdb; pdb.set_trace

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))