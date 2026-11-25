# -*- coding: utf-8 -*-
"""TTS engine adapters for Arabic Voice Studio.

Engines:
- habibi : Habibi-TTS (F5-TTS based, zero-shot voice cloning). Best quality.
           Called through its CLI so we do not depend on internal APIs.
- silma  : SILMA Arabic TTS v1 (F5-TTS based, voice cloning). Python API.
- mock   : test engine (AVS_MOCK=1). Generates a tone, no models needed.

Every engine's synthesize() takes a reference voice (wav + transcript) and the
final (reviewed, diacritized) text, and writes one WAV file.
"""

import os
import re
import subprocess
import wave
import math
import struct
import time
from pathlib import Path

MOCK = os.environ.get("AVS_MOCK") == "1"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- text utils

_SENT_SPLIT = re.compile(r"(?<=[\.!\?؟،؛:\n])\s+")


def chunk_text(text: str, max_chars: int = 250):
    """Split text into sentence-sized chunks that TTS models handle well."""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    chunks, current = [], ""
    for part in parts:
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def concat_wavs(wav_paths, out_path):
    """Concatenate mono WAV files of identical format using only the stdlib."""
    params = None
    frames = []
    for p in wav_paths:
        with wave.open(str(p), "rb") as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    with wave.open(str(out_path), "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)


def _out_path(voice_name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w\-]+", "_", voice_name) or "voice"
    return OUTPUT_DIR / f"{stamp}_{safe}.wav"


# ---------------------------------------------------------------- engines

def synthesize_mock(ref_wav, ref_text, gen_text, out_wav, dialect=None, speed=1.0):
    """Deterministic test engine: a short tone per chunk. Needs no models."""
    seconds = max(1.0, min(4.0, len(gen_text) / 40.0))
    framerate = 24000
    freq = 220.0 if ref_text and "female" in str(ref_wav) else 160.0
    n = int(framerate * seconds)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        for i in range(n):
            sample = int(8000 * math.sin(2 * math.pi * freq * i / framerate))
            w.writeframes(struct.pack("<h", sample))
    return out_wav


def synthesize_habibi(ref_wav, ref_text, gen_text, out_wav, dialect="MSA", speed=1.0):
    """Habibi-TTS via its CLI (installed by install.bat / pip install habibi-tts)."""
    chunk_dir = Path(out_wav).parent / (Path(out_wav).stem + "_chunks")
    chunk_dir.mkdir(exist_ok=True)
    chunk_files = []
    for i, chunk in enumerate(chunk_text(gen_text)):
        chunk_out = chunk_dir / f"chunk_{i:03d}.wav"
        cmd = [
            "habibi-tts_infer-cli",
            "--ref_audio", str(ref_wav),
            "--ref_text", ref_text,
            "--gen_text", chunk,
            "--output_dir", str(chunk_dir),
            "--output_file", chunk_out.name,
        ]
        if dialect:
            cmd += ["--dialect", dialect]
        subprocess.run(cmd, check=True)
        produced = chunk_out if chunk_out.exists() else _newest_wav(chunk_dir)
        chunk_files.append(produced)
    if len(chunk_files) == 1:
        Path(chunk_files[0]).replace(out_wav)
    else:
        concat_wavs(chunk_files, out_wav)
    return out_wav


def synthesize_silma(ref_wav, ref_text, gen_text, out_wav, dialect=None, speed=1.0):
    """SILMA Arabic TTS v1 through its documented Python API (lazy import)."""
    from silma_tts.api import SilmaTTS
    global _silma
    try:
        _silma
    except NameError:
        _silma = SilmaTTS()
    chunks = chunk_text(gen_text)
    if len(chunks) == 1:
        _silma.infer(ref_file=str(ref_wav), ref_text=ref_text,
                     gen_text=chunks[0], file_wave=str(out_wav), seed=None, speed=speed)
    else:
        chunk_dir = Path(out_wav).parent / (Path(out_wav).stem + "_chunks")
        chunk_dir.mkdir(exist_ok=True)
        chunk_files = []
        for i, chunk in enumerate(chunks):
            cf = chunk_dir / f"chunk_{i:03d}.wav"
            _silma.infer(ref_file=str(ref_wav), ref_text=ref_text,
                         gen_text=chunk, file_wave=str(cf), seed=None, speed=speed)
            chunk_files.append(cf)
        concat_wavs(chunk_files, out_wav)
    return out_wav


_ice_model = None


def build_ice_instruct(gender: str, age: str) -> str:
    """Compose the ICE-012 free-text voice instruction (no reference audio needed).

    gender: 'male' | 'female' ; age: 'child' | 'teenager' | 'young adult' | 'middle-aged'
    """
    instruct = f"{gender}, {age}"
    if age in ("child", "teenager"):
        instruct += ", high pitch"
    return instruct


def synthesize_ice(ref_wav, ref_text, gen_text, out_wav, dialect=None, speed=1.0, instruct=None):
    """ICE-012 (darkps/ice-012-audio): text-only TTS with age/gender/style control.

    Loads ~1.4 GB of weights plus the Higgs audio tokenizer on first use.
    """
    global _ice_model
    import soundfile as sf
    import torch
    if _ice_model is None:
        from transformers import AutoModelForCausalLM
        _ice_model = AutoModelForCausalLM.from_pretrained(
            "darkps/ice-012-audio",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    sr = int(getattr(_ice_model, "sampling_rate", 24000) or 24000)
    chunks = chunk_text(gen_text)
    if len(chunks) == 1:
        audio = _ice_model.generate(text=chunks[0], language="arabic",
                                    instruct=instruct, speed=speed)[0]
        sf.write(str(out_wav), audio, sr)
    else:
        chunk_dir = Path(out_wav).parent / (Path(out_wav).stem + "_chunks")
        chunk_dir.mkdir(exist_ok=True)
        chunk_files = []
        for i, chunk in enumerate(chunks):
            cf = chunk_dir / f"chunk_{i:03d}.wav"
            audio = _ice_model.generate(text=chunk, language="arabic",
                                        instruct=instruct, speed=speed)[0]
            sf.write(str(cf), audio, sr)
            chunk_files.append(cf)
        concat_wavs(chunk_files, out_wav)
    return out_wav


def _newest_wav(folder: Path) -> Path:
    wavs = sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if not wavs:
        raise RuntimeError(f"Engine produced no wav in {folder}")
    return wavs[-1]


ENGINES = {
    "habibi": ("Habibi-TTS (أفضل جودة، استنساخ من تسجيل)", synthesize_habibi),
    "silma": ("SILMA (خفيف وسريع، استنساخ من تسجيل)", synthesize_silma),
    "ice": ("ICE-012 (أصوات أطفال ومراهقين بدون تسجيل)", synthesize_ice),
}

TEXT_ONLY_ENGINES = {"ice"}
if MOCK:
    ENGINES["mock"] = ("MOCK (تجربة بدون موديلات)", synthesize_mock)


def synthesize(engine_key, voice, gen_text, dialect, speed, progress=None, instruct=None):
    """voice: dict with wav path + transcript (ignored by text-only engines).
    instruct: free-text voice description for text-only engines (e.g. 'female, child').
    Returns the output wav path."""
    label, fn = ENGINES[engine_key]
    name = voice["name"] if voice else (instruct or "ice").replace(", ", "-")
    out = _out_path(name)
    if engine_key in TEXT_ONLY_ENGINES:
        fn(None, None, gen_text, out, dialect=dialect, speed=speed, instruct=instruct)
    else:
        fn(voice["wav"], voice["txt"], gen_text, out, dialect=dialect, speed=speed)
    return str(out)
