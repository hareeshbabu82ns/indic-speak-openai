import os
import sys
from time import time
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

### Bodhan-AI version
print("Downloading Indic-Speak...")
tts_dir = snapshot_download(repo_id="bodhan-ai/indic-speak", token=hf_token)
sys.path.insert(0, tts_dir)
# Import native loaders from downloaded code
from inference import TTS

print("Indic-Speak downloaded to:", tts_dir)

print("Loading speech checkpoints into memory...")
tts = TTS(tts_dir, device="mps")
# tts = TTS("bodhan-ai/indic-speak")
print("Indic-Speak Speech checkpoints loaded.")

short_text = (
    "కరాగ్రే వసతే లక్ష్మీః కరమధ్యే సరస్వతీ కరమూలే తు గోవిందః ప్రభాతే కర దర్శనమ్‌"
)
# short_text = "కోన్వస్మిన్సామ్ప్రతం లోకే గుణవాన్కశ్చవీర్యవాన్ ధర్మజ్ఞశ్చ కృతజ్ఞశ్చ సత్యవాక్యో దృఢవ్రతః"
long_text = (
    "ఒక ఊరిలో ప్రతి సంవత్సరం దేవుడిని ఊరేగింపు తీసుకెళ్లే సాంప్రదాయం ఉండేది. "
    "ప్రతి సంవత్సరం ఊళ్ళోవారంతా పండగ చేసుకుని, పూజలు చేసి, ఊరేగింపు కోసం అన్ని వీధులు శుభ్రం చేసి, "
    "ముగ్గులతో, తోరణాలతో, పువ్వులతో, చాలా అందంగా అలంకరించేవారు."
)

start_time = time()
wav = tts(
    short_text,
    speaker="Sravani",
)
end_time = time()
print("Time taken:", end_time - start_time)
tts.save("data/out/default-output.wav", wav)  # float32 numpy @ 24 kHz


start_time = time()
wav = tts(
    long_text,
    speaker="Sravani",
)
end_time = time()
print("Time taken for long text:", end_time - start_time)
tts.save("data/out/default-output-long.wav", wav)  # float32 numpy @ 24 kHz

### MLX version
print("Downloading Indic-Speak MLX...")
tts_mlx_dir = snapshot_download("iniquitous/indic-speak-mlx-bf16")
sys.path.insert(0, tts_mlx_dir)
from tts_mlx import MLXTTS, speak_long, stream_long

print("Indic-Speak MLX downloaded to:", tts_mlx_dir)

mlx_tts = MLXTTS(tts_mlx_dir)
print("Indic-Speak MLX Speech checkpoints loaded.")

# single sentence
start_time = time()
wav, stats = mlx_tts(short_text, speaker="Sravani")
end_time = time()
sf.write("data/out/mlx-output.wav", wav, mlx_tts.sample_rate)
print(stats)  # tokens, tok_per_s, ttft, rtf ...
print("Time taken for single sentence:", end_time - start_time)

# multi-sentence text (recommended for quantized variants — see caveats)
start_time = time()
wav, per_sentence = speak_long(mlx_tts, long_text, speaker="Sravani")
end_time = time()

sf.write("data/out/mlx-out-long.wav", wav, mlx_tts.sample_rate)
print(per_sentence)  # tokens, tok_per_s, ttft, rtf ...
print("Time taken for long text (MLX):", end_time - start_time)

# # streaming: yields float32 chunks (~0.7 s) while generation is still running
# for chunk, st in stream_long(mlx_tts, long_text, speaker="Sravani"):
#     play(chunk)   # your audio sink

# Download Speech Repositories
print("Downloading Indic-Transcribe-Core...")
asr_dir = snapshot_download(repo_id="bodhan-ai/indic-transcribe-core", token=hf_token)
sys.path.insert(0, asr_dir)
print("Indic-Transcribe-Core downloaded to:", asr_dir)

from indic_transcribe import IndicTranscribe

asr = IndicTranscribe.from_pretrained(asr_dir, device="mps")
start_time = time()
print(asr("data/out/default-output.wav", lang="sa"))
end_time = time()
print("Time taken for ASR:", end_time - start_time)

from long_form import transcribe_long

start_time = time()
text = transcribe_long(asr, "data/out/default-output-long.wav", lang=None)
end_time = time()
print(text)
print("Time taken for long ASR:", end_time - start_time)
