"""End-to-end verification suite for the AI Serving Platform (phase 1: TTS + ASR).

Run against a live server:

    ./start.sh                      # terminal 1
    python test_api.py              # terminal 2
    python test_api.py --base-url http://127.0.0.1:8000 --skip-lifecycle

Files and models created by the suite are removed again at the end.
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import time
import traceback

import numpy as np
import requests
import soundfile as sf

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 900          # first requests may download and load multi-GB models
HINDI = "नमस्ते, आज हम विज्ञान पढ़ेंगे। प्रकाश की चाल लगभग तीन लाख किलोमीटर प्रति सेकंड होती है।"
TELUGU = "నమస్కారం, ఇది ప్రత్యక్ష ధ్వని ప్రసార పరీక్ష. రెండవ వాక్యం కూడా ఉంది."
TEST_MODEL_ID = "test-whisper-tiny"

created_files: list[str] = []
shared: dict = {}


def step(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def get(path: str, **kw) -> requests.Response:
    return requests.get(BASE_URL + path, timeout=TIMEOUT, **kw)


def post(path: str, **kw) -> requests.Response:
    return requests.post(BASE_URL + path, timeout=TIMEOUT, **kw)


def speech(**payload) -> requests.Response:
    res = post("/v1/audio/speech", json=payload, stream=bool(payload.get("stream")))
    if res.headers.get("X-File-Id"):
        created_files.append(res.headers["X-File-Id"])
    return res


def transcribe(audio: bytes, endpoint: str = "transcriptions", filename: str = "test.wav", **data) -> requests.Response:
    res = post(f"/v1/audio/{endpoint}", files={"file": (filename, audio, "audio/wav")}, data=data)
    if res.headers.get("X-File-Id"):
        created_files.append(res.headers["X-File-Id"])
    return res


def assert_error(res: requests.Response, status: int) -> None:
    assert res.status_code == status, f"expected {status}, got {res.status_code}: {res.text[:200]}"
    err = res.json()["error"]
    assert err["message"] and err["type"], f"not an OpenAI-style error body: {res.text[:200]}"


# ----------------------------------------------------------------------------
def test_system():
    step("1. System & health")
    health = get("/health").json()
    assert health["status"] == "ok" and health["sample_rates"] == {"tts": 24000, "asr": 16000}
    print(f"health: device={health['device']} loaded={health['loaded_models']} defaults={health['defaults']}")

    status = get("/v1/system/status")
    assert status.status_code == 200
    s = status.json()
    assert s["memory"]["total_gb"] > 0 and s["memory"]["free_gb"] >= 0
    assert "chip" in s["device"] and "hf_cache_gb" in s["disk"] and isinstance(s["loaded_models"], list)
    print(f"system: {s['device']['chip']} | {s['memory']['used_gb']}/{s['memory']['total_gb']} GB | "
          f"mlx={s['mlx']['available']} mps={s['mps']['available']} | cache {s['disk']['hf_cache_gb']} GB")

    models = get("/v1/models").json()
    assert models["object"] == "list"
    by_id = {m["id"]: m for m in models["data"]}
    print("models:", ", ".join(f"{m['id']}[{m['modality']}/{m['status']}]" for m in models["data"]))
    for required in ("indic-speak-torch", "indic-transcribe", "tts-1", "tts-1-hd", "whisper-1"):
        assert required in by_id, f"{required} missing from /v1/models"
    assert "indic-speak-mlx" in by_id or platform.machine() != "arm64"
    for m in models["data"]:
        assert m["object"] == "model" and m["modality"] in ("tts", "asr", "llm", "vlm", "embedding", "rerank", "image_gen")
        assert {"backend", "repo_id", "status", "size_mb"} <= set(m), f"extended fields missing on {m['id']}"
    assert by_id["tts-1"]["alias_of"] == health["defaults"]["tts"]
    assert by_id["whisper-1"]["alias_of"] == health["defaults"]["asr"]

    one = get("/v1/models/indic-transcribe").json()
    assert one["id"] == "indic-transcribe" and one["modality"] == "asr"
    assert_error(get("/v1/models/does-not-exist"), 404)
    assert get("/").status_code == 200 and "<html" in get("/ui").text.lower()
    print("OK")


def test_catalogs():
    step("2. Voice and language catalogs")
    voices = get("/v1/audio/voices", params={"model": "indic-speak-torch"}).json()
    assert len(voices["voices"]) == 44, f"expected 44 voices, got {len(voices['voices'])}"
    assert len(voices["by_language"]) == 22
    for alias, target in {"alloy": "Amit", "echo": "Arun", "fable": "Deepika", "onyx": "Vamsi", "nova": "Kavya", "shimmer": "Sravani"}.items():
        assert voices["openai_aliases"][alias] == target
    assert "NEUTRAL" in voices["styles"] and "educational lecture" in voices["styles"]
    assert all({"gender", "lang", "lang_name"} <= set(v) for v in voices["voices"].values())
    print(f"voices: {len(voices['voices'])} across {len(voices['by_language'])} languages; models: {list(voices['models'])}")

    langs = get("/v1/audio/languages", params={"model": "indic-transcribe"}).json()
    assert len(langs["languages"]) == 28 and "auto" in langs["languages"] and "te" in langs["languages"]
    assert langs["modes"] == ["native", "mixed", "romanised"]
    print(f"languages: {len(langs['languages'])} (incl. auto); modes: {langs['modes']}")
    print("OK")


def test_model_lifecycle():
    step("3. Model management lifecycle (inspect -> register -> download -> load -> infer -> unload -> delete)")
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    repo = "mlx-community/whisper-tiny" if apple else "openai/whisper-tiny"

    info = get("/v1/hf/inspect", params={"repo_id": repo})
    assert info.status_code == 200, info.text
    info = info.json()
    assert info["suggested"]["modality"] == "asr" and info["size_mb"] > 10 and info["supported"]
    print(f"inspect: {info['repo_id']} {info['size_mb']} MB -> {info['suggested']}")
    assert_error(get("/v1/hf/inspect", params={"repo_id": "no-such-org/no-such-model-xyz"}), 404)

    requests.delete(f"{BASE_URL}/v1/models/{TEST_MODEL_ID}", timeout=TIMEOUT)       # leftover from an aborted run
    reg = post("/v1/models/register", json={"id": TEST_MODEL_ID, "repo_id": repo, "modality": "asr",
                                            "backend": info["suggested"]["backend"]})
    assert reg.status_code == 200, reg.text
    assert reg.json()["id"] == TEST_MODEL_ID and reg.json()["status"] in ("registered", "cached")
    assert_error(post("/v1/models/register", json={"id": TEST_MODEL_ID, "repo_id": repo, "modality": "asr"}), 409)
    assert_error(post("/v1/models/register", json={"repo_id": repo, "modality": "telepathy"}), 400)
    assert get(f"/v1/models/{TEST_MODEL_ID}").json()["repo_id"] == repo

    task = post(f"/v1/models/{TEST_MODEL_ID}/download").json()
    assert task["task_id"] and task["status"] in ("downloading", "completed")
    while True:
        t = get(f"/v1/models/tasks/{task['task_id']}").json()
        print(f"  download {t['status']} {t['progress']:.0f}% ({t['downloaded_bytes']}/{t['total_bytes']} bytes)")
        if t["status"] != "in_progress":
            break
        time.sleep(1)
    assert t["status"] == "completed", t
    assert get(f"/v1/models/{TEST_MODEL_ID}").json()["status"] == "cached"

    loaded = post(f"/v1/models/{TEST_MODEL_ID}/load")
    assert loaded.status_code == 200 and loaded.json()["status"] == "loaded", loaded.text
    assert TEST_MODEL_ID in [m["id"] for m in get("/v1/system/status").json()["loaded_models"]]
    print(f"loaded in {loaded.json()['load_seconds']}s")

    tone = np.zeros(16000, dtype=np.float32)        # 1 s of silence: just proves the adapter runs
    buf = io.BytesIO(); sf.write(buf, tone, 16000, format="WAV")
    res = transcribe(buf.getvalue(), model=TEST_MODEL_ID, response_format="verbose_json")
    assert res.status_code == 200 and "text" in res.json(), res.text

    unloaded = post(f"/v1/models/{TEST_MODEL_ID}/unload").json()
    assert unloaded["unloaded"] is True and unloaded["status"] == "cached"
    deleted = requests.delete(f"{BASE_URL}/v1/models/{TEST_MODEL_ID}", params={"delete_files": "false"}, timeout=TIMEOUT)
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert_error(get(f"/v1/models/{TEST_MODEL_ID}"), 404)
    print("OK")


def test_tts():
    step("4. Text-to-Speech: non-streaming")
    t0 = time.time()
    res = speech(model="tts-1", input=HINDI, voice="alloy", response_format="wav")
    assert res.status_code == 200, res.text
    assert res.headers["Content-Type"].startswith("audio/wav")
    file_id, file_url = res.headers.get("X-File-Id"), res.headers.get("X-File-Url")
    assert file_id and file_url == f"/v1/audio/files/{file_id}"
    assert "attachment" in res.headers["Content-Disposition"]
    data, sr = sf.read(io.BytesIO(res.content))
    assert sr == 24000 and data.ndim == 1 and len(data) > sr, f"bad audio: sr={sr} samples={len(data)}"
    assert abs(float(res.headers["X-Audio-Duration"]) - len(data) / sr) < 0.05
    assert res.headers["X-Voice"] == "Amit", "OpenAI alias 'alloy' should map to Amit"
    print(f"wav: {len(data) / sr:.2f}s audio in {time.time() - t0:.1f}s (rtf {res.headers['X-RTF']}, model {res.headers['X-Model']})")
    shared["hindi_wav"] = res.content

    dl = get(f"/v1/audio/files/{file_id}/download")
    assert dl.status_code == 200 and dl.content == res.content and "attachment" in dl.headers["Content-Disposition"]
    assert get(file_url).content == res.content
    print("download round trip matches")

    base_len = None
    for fmt, mime in (("mp3", "audio/mpeg"), ("flac", "audio/flac"), ("pcm", "audio/pcm")):
        r = speech(input="नमस्ते, यह एक परीक्षण है।", voice="Kavya", response_format=fmt, seed=11, save=False)
        assert r.status_code == 200 and r.headers["Content-Type"].startswith(mime), f"{fmt}: {r.text[:200]}"
        assert "X-File-Id" not in r.headers, "save=false must not store the file"
        if fmt == "pcm":
            assert len(r.content) % 2 == 0 and len(r.content) > 24000
        else:
            d, s = sf.read(io.BytesIO(r.content))
            assert s == 24000 and len(d) > 12000
            base_len = base_len or len(d)
        print(f"{fmt}: {len(r.content)} bytes")

    fast = speech(input="नमस्ते, यह एक परीक्षण है।", voice="Kavya", response_format="wav", speed=2.0, seed=11, save=False)
    assert fast.status_code == 200
    d, _ = sf.read(io.BytesIO(fast.content))
    print(f"speed 2.0x: {len(d) / 24000:.2f}s vs {base_len / 24000:.2f}s at 1.0x")
    assert len(d) < 0.8 * base_len, "speed=2.0 should shorten the audio"

    assert_error(speech(input="hi", response_format="aiff"), 400)
    assert_error(speech(input="hi", speed=9), 400)
    assert_error(speech(input="   "), 400)
    assert_error(speech(input="hi", model="no-such-model"), 404)
    assert_error(speech(input="hi", model="indic-transcribe"), 400)
    print("OK")


def test_tts_streaming():
    step("5. Text-to-Speech: streaming")
    t0, ttfa, chunks, body = time.time(), None, 0, b""
    with speech(input=TELUGU, voice="Sravani", response_format="wav", stream=True) as res:
        assert res.status_code == 200, res.text
        assert res.headers.get("Transfer-Encoding") == "chunked"
        file_id = res.headers["X-File-Id"]
        for chunk in res.iter_content(chunk_size=None):
            if chunk:
                if len(body) <= 44 < len(body) + len(chunk) and ttfa is None:
                    ttfa = time.time() - t0
                chunks += 1
                body += chunk
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE" and body[36:40] == b"data"
    assert body[4:8] == b"\xff\xff\xff\xff" and body[40:44] == b"\xff\xff\xff\xff", "streaming WAV header must be indefinite"
    seconds = (len(body) - 44) / 2 / 24000
    assert chunks > 2 and seconds > 1.0
    print(f"wav stream: {chunks} chunks, {seconds:.2f}s audio, first audio after {ttfa:.2f}s, total {time.time() - t0:.2f}s")
    shared["telugu_pcm"] = body[44:]

    time.sleep(0.5)     # the stream is persisted right after the last chunk
    saved = get(f"/v1/audio/files/{file_id}/download")
    assert saved.status_code == 200
    d, sr = sf.read(io.BytesIO(saved.content))
    assert sr == 24000 and abs(len(d) / sr - seconds) < 0.1, "saved stream should match what was sent"

    with speech(input="नमस्ते।", response_format="mp3", stream=True, save=False) as res:
        mp3 = b"".join(res.iter_content(chunk_size=None))
    d, sr = sf.read(io.BytesIO(mp3))
    assert sr == 24000 and len(d) > 6000
    print(f"mp3 stream: {len(mp3)} bytes, {len(d) / sr:.2f}s")
    assert_error(speech(input="hi", response_format="flac", stream=True), 400)
    print("OK")


def test_asr():
    step("6. Speech-to-Text")
    audio = shared["hindi_wav"]
    res = transcribe(audio, model="whisper-1", response_format="json")
    assert res.status_code == 200, res.text
    text = res.json()["text"]
    assert len(text) > 10 and res.headers["X-File-Id"]
    assert res.headers["X-Language"] in get("/v1/audio/languages").json()["languages"]
    print(f"json (auto language -> {res.headers['X-Language']}): {text}")
    assert "विज्ञान" in text or "प्रकाश" in text, "transcript should recover words of the synthesised sentence"

    plain = transcribe(audio, language="hi", response_format="text", save="false")
    assert plain.status_code == 200 and plain.headers["Content-Type"].startswith("text/plain")
    assert plain.text.strip() and not plain.text.lstrip().startswith("{") and "X-File-Id" not in plain.headers

    # long-form: > 30 s must be cut at pauses into several timestamped segments
    pcm = np.frombuffer(shared["telugu_pcm"], dtype="<i2").astype(np.float32) / 32768
    gap = np.zeros(int(0.6 * 24000), np.float32)
    long_wav = np.concatenate([np.concatenate([pcm, gap])] * (int(40 * 24000 / (len(pcm) + len(gap))) + 1))
    buf = io.BytesIO(); sf.write(buf, long_wav, 24000, format="FLAC")
    verbose = transcribe(buf.getvalue(), filename="long.flac", language="auto", response_format="verbose_json")
    assert verbose.status_code == 200, verbose.text
    v = verbose.json()
    assert v["task"] == "transcribe" and v["language"] and abs(v["duration"] - len(long_wav) / 24000) < 0.2
    segs = v["segments"]
    assert len(segs) >= 2, f"{v['duration']}s audio should produce several segments, got {len(segs)}"
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"] + 1e-6 and a["start"] < a["end"]
    assert max(s["end"] - s["start"] for s in segs) <= 30.01 and abs(segs[-1]["end"] - v["duration"]) < 0.5
    assert all({"id", "start", "end", "text"} <= set(s) for s in segs)
    print(f"verbose_json: {v['duration']}s, language={v['language']}, {len(segs)} segments, "
          f"first [{segs[0]['start']} -> {segs[0]['end']}] {segs[0]['text'][:50]}")

    srt = transcribe(audio, language="hi", response_format="srt", save="false")
    assert srt.status_code == 200 and srt.text.startswith("1\n00:00:00,000 --> ")
    vtt = transcribe(audio, language="hi", response_format="vtt", save="false")
    assert vtt.status_code == 200 and vtt.text.startswith("WEBVTT") and "00:00:00.000 --> " in vtt.text
    print("srt:", srt.text.strip().splitlines()[1])

    roman = transcribe(audio, endpoint="translations", response_format="json", save="false")
    assert roman.status_code == 200, roman.text
    rt = roman.json()["text"]
    latin = sum(c.isascii() or "Ā" <= c <= "ỿ" for c in rt) / max(len(rt), 1)
    print(f"translations ({roman.headers.get('X-Script-Mode')}): {rt}")
    assert latin > 0.9, "translations should return Latin-script text"

    assert_error(transcribe(b"this is not audio", save="false"), 400)
    assert_error(transcribe(audio, response_format="docx"), 400)
    assert_error(transcribe(audio, model="indic-speak-torch"), 400)
    print("OK")


def test_files():
    step("7. Saved files")
    files = get("/v1/audio/files").json()
    ids = {f["id"] for f in files}
    assert set(created_files) <= ids, "files created by this run should be listed"
    tts = next(f for f in files if f["type"] == "tts" and f["id"] in created_files)
    asr = next(f for f in files if f["type"] == "asr" and f["id"] in created_files)
    assert {"model", "voice", "duration", "text", "audio_url", "download_url", "created_at"} <= set(tts)
    assert {"model", "language", "duration", "text", "segments"} <= set(asr)
    assert all(f["type"] == "asr" for f in get("/v1/audio/files", params={"type": "asr"}).json())
    print(f"{len(files)} files on the server, {len(created_files)} from this run")

    assert get(f"/v1/audio/files/{asr['id']}").json()["text"] == asr["text"]
    assert get(f"/v1/audio/files/{asr['id']}", params={"audio": "true"}).headers["Content-Type"].startswith("audio/")
    assert "-->" in get(f"/v1/audio/files/{asr['id']}/download", params={"format": "srt"}).text
    assert_error(get("/v1/audio/files/no_such_file"), 404)
    assert_error(get("/v1/audio/files/..%2F..%2Fserver.py"), 404)

    victim = created_files.pop()
    res = requests.delete(f"{BASE_URL}/v1/audio/files/{victim}", timeout=TIMEOUT)
    assert res.status_code == 200 and res.json()["deleted"] is True
    assert_error(get(f"/v1/audio/files/{victim}"), 404)
    assert_error(requests.delete(f"{BASE_URL}/v1/audio/files/{victim}", timeout=TIMEOUT), 404)
    print("OK")


def cleanup():
    for file_id in created_files:
        requests.delete(f"{BASE_URL}/v1/audio/files/{file_id}", timeout=60)
    requests.delete(f"{BASE_URL}/v1/models/{TEST_MODEL_ID}", timeout=60)


def main() -> int:
    global BASE_URL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--skip-lifecycle", action="store_true", help="skip the register/download/load test (needs internet, ~75 MB)")
    args = parser.parse_args()
    BASE_URL = args.base_url.rstrip("/")

    try:
        requests.get(BASE_URL + "/health", timeout=5)
    except requests.RequestException as e:
        print(f"Server not reachable at {BASE_URL}: {e}\nStart it with ./start.sh first.")
        return 2

    tests = [test_system, test_catalogs, test_model_lifecycle, test_tts, test_tts_streaming, test_asr, test_files]
    if args.skip_lifecycle:
        tests.remove(test_model_lifecycle)
    failed = []
    t0 = time.time()
    try:
        for test in tests:
            try:
                test()
            except Exception:
                failed.append(test.__name__)
                print(f"\nFAILED: {test.__name__}")
                traceback.print_exc()
    finally:
        cleanup()
    print("\n" + "=" * 70)
    print(f"{len(tests) - len(failed)}/{len(tests)} test groups passed in {time.time() - t0:.0f}s" + (f" | failed: {', '.join(failed)}" if failed else ""))
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
