import os
import sys
from dotenv import load_dotenv
import gradio as gr
import torch
import numpy as np
import spaces
from huggingface_hub import snapshot_download
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# --- STEP 1: Download & Initialize Repositories ---
hf_token = os.environ.get("HF_TOKEN")

# 1. Download Speech Repositories
print("Downloading Indic-Transcribe-Core...")
asr_dir = snapshot_download(repo_id="bodhan-ai/indic-transcribe-core", token=hf_token)
sys.path.insert(0, asr_dir)

print("Downloading Indic-Speak...")
tts_dir = snapshot_download(repo_id="bodhan-ai/indic-speak", token=hf_token)
sys.path.insert(0, tts_dir)

# Import native loaders from downloaded code
from indic_transcribe import IndicTranscribe
from inference import TTS

print("Loading speech checkpoints into memory...")
asr_model = IndicTranscribe.from_pretrained(asr_dir, device="mps")
tts_model = TTS(tts_dir, device="mps")

# 2. Initialize Conversation Layer (Gemma 4 12B IT - Quantized for ZeroGPU)
print("Loading Google Gemma 4 12B conversational model...")
model_id = "google/gemma-4-12B-it"

# 4-bit configuration to make 12B parameters fit onto dynamic space slots safely
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=(
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ),
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id, quantization_config=quantization_config, low_cpu_mem_usage=True, device="mps"
)


# --- STEP 2: Unified Voice-to-Voice Conversation Pipeline (Batch Mode) ---
@spaces.GPU(duration=60)  # Allocates GPU on-demand when the button is clicked
def voice_to_voice_pipeline(audio_input, language_code, speaker_name, emotion_style):
    if audio_input is None:
        return "Please provide an audio recording.", "", None

    try:
        # --- PHASE 1: ASR (Indic-Transcribe-Core) ---
        lang_param = None if language_code == "Auto-detect" else language_code
        user_prompt = asr_model(audio_input, lang=lang_param).strip()

        if not user_prompt:
            return "Could not resolve speech into text.", "", None

        # --- PHASE 2: Conversational LLM Layer (Gemma 4 12B) ---
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. Answer the user's query directly and concisely. "
                    "You MUST reply using the exact same language and script that the user used. "
                    "Keep your response under 3 sentences so it sounds natural when spoken aloud. "
                    "Do not add any markdown formatting like bold asterisks or bullet points."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        # Format via chat template and prepare tensor maps
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Generate raw response tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.4, do_sample=True
            )

        # Isolate new tokens and decode
        generated_tokens = outputs[0][inputs.input_ids.shape[-1] :]
        gemma_response = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        # --- PHASE 3: TTS (Indic-Speak) ---
        wav_array = tts_model(
            text=gemma_response,
            speaker=speaker_name,
            style=emotion_style,
            temperature=0.6,
        )

        # --- AUDIO POST-PROCESSING & SAFE INT16 CASTING ---
        if hasattr(wav_array, "cpu"):
            wav_array = wav_array.cpu().numpy()
        elif hasattr(wav_array, "numpy"):
            wav_array = wav_array.numpy()

        wav_array = np.squeeze(wav_array)

        max_val = np.max(np.abs(wav_array))
        if max_val > 0:
            wav_array = wav_array / max_val

        audio_int16 = (wav_array * 32767).astype(np.int16)

        return user_prompt, gemma_response, (24000, audio_int16)

    except Exception as e:
        return f"Pipeline execution failed: {str(e)}", "", None


# --- STEP 3: Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Voice-In, Voice-Out AI Assistant (Gemma 4 12B)")
    gr.Markdown(
        "Record your voice, click **Process Voice Message**, and the assistant will answer you back out loud using a quantized Gemma 4 12B deployment."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎛️ Settings")
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your Voice Input",
            )

            lang_dropdown = gr.Dropdown(
                choices=[
                    "Auto-detect",
                    "en",
                    "hi",
                    "bn",
                    "mr",
                    "te",
                    "ta",
                    "gu",
                    "kn",
                    "ml",
                    "or",
                    "pa",
                    "as",
                    "ur",
                    "bho",
                    "brx",
                    "doi",
                    "kok",
                    "ks",
                    "mai",
                    "ne",
                    "mni",
                    "sa",
                    "sat",
                    "sd",
                ],
                value="Auto-detect",
                label="Input Language Script Profile",
            )

            speaker_dropdown = gr.Dropdown(
                choices=[
                    "Amit",
                    "Kavya",
                    "Adarsh",
                    "Deepika",
                    "Sourav",
                    "Ishita",
                    "Arun",
                    "Anitha",
                ],
                value="Amit",
                label="AI Voice Speaker Profile",
            )

            style_dropdown = gr.Dropdown(
                choices=["NEUTRAL", "ANGER", "FEAR", "HAPPY", "SAD"],
                value="NEUTRAL",
                label="AI Vocal Emotion",
            )

            submit_btn = gr.Button("Process Voice Message", variant="primary")

        with gr.Column():
            gr.Markdown("### 📻 Conversation Tracking")
            text_raw = gr.Textbox(label="1. What You Said (ASR)", interactive=False)
            text_clean = gr.Textbox(
                label="2. AI's Text Reply (Gemma 4 Response)", interactive=False
            )
            audio_out = gr.Audio(label="3. AI's Voice Reply (Indic-Speak)")

    submit_btn.click(
        fn=voice_to_voice_pipeline,
        inputs=[audio_in, lang_dropdown, speaker_dropdown, style_dropdown],
        outputs=[text_raw, text_clean, audio_out],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base())
