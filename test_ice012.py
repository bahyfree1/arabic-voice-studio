# -*- coding: utf-8 -*-
"""ICE-012 one-click test: generates child voice samples so we can hear the quality.
Run test_ice012.bat - it calls this script. Outputs wav+mp3 files in outputs/."""
import sys, os, subprocess
from pathlib import Path

LESSON = """لِمَاذَا تَبْدُو السَّمَاءُ زَرْقَاءَ؟ هَلْ نَظَرْتَ يَوْمًا إِلَى السَّمَاءِ، وَتَسَاءَلْتَ: لِمَاذَا لَوْنُهَا أَزْرَقُ؟"""

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

def to_mp3(wav_path):
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        mp3 = str(wav_path).replace(".wav", ".mp3")
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(wav_path),
                        "-codec:a", "libmp3lame", "-q:a", "4", mp3], check=True)
        return mp3
    except Exception as e:
        print("mp3 conversion skipped (", e, ") - the wav file is fine to send")
        return None

print("Loading ICE-012 (first run downloads ~1.4 GB + the audio tokenizer)...")
import torch
from transformers import AutoModelForCausalLM

if torch.cuda.is_available():
    dtype = torch.float16
    print("GPU found:", torch.cuda.get_device_name(0), "- using fp16 on GPU")
else:
    dtype = torch.float32
    print("No GPU found - using CPU (slower)")

model = AutoModelForCausalLM.from_pretrained(
    "darkps/ice-012-audio",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=dtype,
)
sr = int(getattr(model, "sampling_rate", 24000) or 24000)
import soundfile as sf

TESTS = [
    ("ice_girl_child", "female, child, high pitch"),
    ("ice_boy_child", "male, child, high pitch"),
    ("ice_girl_child_b", "female, child, high pitch, playful"),
    ("ice_boy_teen", "male, teenager"),
]

for name, instruct in TESTS:
    print("Generating", name, "->", instruct)
    try:
        audio = model.generate(text=LESSON, language="arabic", instruct=instruct)[0]
        wav = OUT / (name + ".wav")
        sf.write(str(wav), audio, sr)
        mp3 = to_mp3(wav)
        print("  saved:", wav.name, "(%.1fs)" % (len(audio) / sr), ("+ mp3" if mp3 else ""))
    except Exception as e:
        print("  FAILED:", e)

print()
print("Done. Files are in the outputs folder - send the mp3 files back on WhatsApp.")
