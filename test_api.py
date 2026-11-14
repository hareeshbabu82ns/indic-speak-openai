"""Comprehensive test suite for OpenAI-compatible Indic-Speak TTS & ASR API
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
import soundfile as sf

try:
    import requests
except ImportError:
    import urllib.request as urllib_req
    import urllib.error as urllib_err

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def print_step(title: str):
    print("\n" + "=" * 60)
    print(f"👉 {title}")
    print("=" * 60)


def test_health():
    print_step("Test 1: Health & Models Check")
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health = res.json()
    print("Health response:", json.dumps(health, indent=2))
    assert health["status"] == "ok"

    res_models = requests.get(f"{BASE_URL}/v1/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    model_ids = [m["id"] for m in models_data["data"]]
    print("Available models:", model_ids)
    assert "indic-speak-mlx" in model_ids or "indic-speak-torch" in model_ids
    assert "indic-transcribe" in model_ids
    assert "tts-1" in model_ids
    assert "whisper-1" in model_ids
    print("✅ Health & Models endpoints verified!")


def test_voices_and_languages():
    print_step("Test 2: Voices and Languages Metadata")
    res = requests.get(f"{BASE_URL}/v1/audio/voices")
    assert res.status_code == 200
    voices = res.json()
    print(f"Loaded {len(voices['voices'])} voices across 22 Indic languages.")
    assert "Sravani" in voices["voices"]
    assert "Amit" in voices["voices"]
    assert "alloy" in voices["openai_aliases"]

    res_langs = requests.get(f"{BASE_URL}/v1/audio/languages")
    assert res_langs.status_code == 200
    langs = res_langs.json()
    print(f"Loaded {len(langs['languages'])} ASR languages.")
    assert "te" in langs["languages"]
    assert "hi" in langs["languages"]
    print("✅ Voices and Languages catalog verified!")


def test_tts_non_streaming():
    print_step("Test 3: OpenAI TTS (/v1/audio/speech) - Non-streaming & Save")
    payload = {
        "model": "indic-speak-mlx",
        "input": "కరాగ్రే వసతే లక్ష్మీః",
        "voice": "Sravani",
        "response_format": "wav",
        "save": True
    }
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
    elapsed = time.time() - t0
    assert res.status_code == 200, f"TTS request failed: {res.text}"

    content_type = res.headers.get("Content-Type", "")
    assert "audio/wav" in content_type or "audio/" in content_type
    file_id = res.headers.get("X-File-Id")
    file_url = res.headers.get("X-File-Url")
    audio_duration = res.headers.get("X-Audio-Duration")

    print(f"Synthesized in {elapsed:.2f}s | Audio duration: {audio_duration}s")
    print(f"Headers: X-File-Id={file_id}, X-File-Url={file_url}")
    assert file_id is not None
    assert len(res.content) > 1000

    # Verify audio content can be parsed by soundfile
    data, sr = sf.read(io.BytesIO(res.content))
    print(f"Decoded audio samples: {len(data)}, Sample rate: {sr} Hz")
    assert sr == 24000
    assert len(data) > 0

    # Test downloading the saved file from the server
    print_step("Test 3b: Download Generated Audio File")
    download_res = requests.get(f"{BASE_URL}/v1/audio/files/{file_id}/download")
    assert download_res.status_code == 200
    assert len(download_res.content) == len(res.content)
    print(f"✅ Successfully downloaded {file_id}.wav ({len(download_res.content)} bytes)")


def test_tts_streaming():
    print_step("Test 4: OpenAI TTS (/v1/audio/speech) - Streaming Direct to Client")
    payload = {
        "model": "indic-speak-mlx",
        "input": "నమస్కారం, ఇది ప్రత్యక్ష ధ్వని ప్రసార పరీక్ష.",
        "voice": "Sravani",
        "response_format": "wav",
        "stream": True,
        "save": True
    }

    t0 = time.time()
    ttfa = None
    chunks_received = 0
    total_bytes = 0

    with requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, stream=True) as res:
        assert res.status_code == 200, f"Streaming failed: {res.text}"
        for chunk in res.iter_content(chunk_size=4096):
            if chunk:
                if ttfa is None:
                    ttfa = time.time() - t0
                    print(f"Time to first audio (TTFA): {ttfa:.2f}s")
                chunks_received += 1
                total_bytes += len(chunk)

    total_time = time.time() - t0
    print(f"Stream completed in {total_time:.2f}s: {chunks_received} chunks, {total_bytes} bytes")
    assert chunks_received > 0
    assert total_bytes > 1000
    print("✅ TTS Streaming directly to client verified!")


def test_asr_short_and_long():
    print_step("Test 5: OpenAI ASR (/v1/audio/transcriptions) - Short & Long Audio")

    # 1. Short audio test with default-output.wav
    sample_file = "data/out/default-output.wav"
    if os.path.exists(sample_file):
        print(f"Testing with {sample_file}...")
        with open(sample_file, "rb") as f:
            res = requests.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                files={"file": (os.path.basename(sample_file), f, "audio/wav")},
                data={
                    "model": "indic-transcribe",
                    "language": "sa",
                    "response_format": "json"
                }
            )
        assert res.status_code == 200, f"ASR failed: {res.text}"
        json_resp = res.json()
        print("Transcription (JSON):", json_resp)
        assert "text" in json_resp
        assert len(json_resp["text"]) > 0

    # 2. Long audio test with verbose_json (segments & timestamps)
    long_sample = "data/out/default-output-long.wav"
    if os.path.exists(long_sample):
        print(f"Testing verbose_json with {long_sample}...")
        with open(long_sample, "rb") as f:
            res_verbose = requests.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                files={"file": (os.path.basename(long_sample), f, "audio/wav")},
                data={
                    "model": "indic-transcribe",
                    "language": "auto",  # auto-detect
                    "response_format": "verbose_json"
                }
            )
        assert res_verbose.status_code == 200
        verbose_data = res_verbose.json()
        print("Detected language:", verbose_data.get("language"))
        print("Duration:", verbose_data.get("duration"), "s")
        print("Full transcript:", verbose_data.get("text"))
        print(f"Generated {len(verbose_data.get('segments', []))} timestamped segments.")
        assert verbose_data.get("language") in ("te", "sa", "hi")
        assert len(verbose_data.get("segments", [])) > 0
        seg0 = verbose_data["segments"][0]
        print(f"Segment 0: [{seg0['start']}s -> {seg0['end']}s] {seg0['text']}")

    # 3. Test SRT format output
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            res_srt = requests.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                files={"file": (os.path.basename(sample_file), f, "audio/wav")},
                data={
                    "model": "indic-transcribe",
                    "language": "sa",
                    "response_format": "srt"
                }
            )
        assert res_srt.status_code == 200
        print("SRT output:\n" + res_srt.text)
        assert "-->" in res_srt.text

    print("✅ ASR transcription (short, long, segments, SRT) verified!")


def test_files_listing():
    print_step("Test 6: List Saved Files")
    res = requests.get(f"{BASE_URL}/v1/audio/files")
    assert res.status_code == 200
    files = res.json()
    print(f"Found {len(files)} saved files on server.")
    for f in files[:5]:
        print(f" - [{f.get('type')}] {f.get('id')}: {f.get('filename')} ({f.get('duration')}s)")
    assert len(files) > 0
    print("✅ File listing verified!")


def main():
    print("Starting Indic-Speak & Transcribe API Test Suite...")
    try:
        test_health()
        test_voices_and_languages()
        test_tts_non_streaming()
        test_tts_streaming()
        test_asr_short_and_long()
        test_files_listing()
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🚀")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
