# -*- coding: utf-8 -*-
"""Arabic Voice Studio - public demo (Google Colab, free CPU).

ElevenLabs-style flow: paste Arabic text, pick a voice, press generate, listen.
Runs SILMA TTS (Arabic voice cloning) on CPU.
"""

from pathlib import Path
import re
import time
import tempfile

import gradio as gr

VOICES_DIR = Path(__file__).resolve().parent / "voices"
MAX_CHARS = 1200

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        import torch
        _orig_load = torch.load

        def patched_load(*a, **k):
            if k.get("weights_only") and "mmap" not in k:
                k["mmap"] = True
            return _orig_load(*a, **k)

        torch.load = patched_load
        from silma_tts.api import SilmaTTS
        _tts = SilmaTTS(device="cpu", enable_normalizer=False, force_tashkeel=False)
    return _tts


def list_voices():
    voices = {}
    for wav in [VOICES_DIR / f"{n}.wav" for n in ["female-ar", "child-girl-6", "child-boy-6"]]:
        txt = wav.with_suffix(".txt")
        if txt.exists():
            voices[wav.stem] = {"wav": str(wav),
                                "txt": txt.read_text(encoding="utf-8").strip()}
    return voices


VOICE_LABELS = {
    "female-ar": "ست (عربية فصحى)",
    "child-girl-6": "بنت (٦ سنين)",
    "child-boy-6": "ولد (٧ سنين)",
}


def label_for(name):
    return VOICE_LABELS.get(name, name)


def auto_tashkeel(text):
    from tashkeel_utils import vocalize
    return vocalize(text)


def generate(text, voice_name, speed, seed, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("اكتب نص عربي الأول")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise gr.Error(f"النص طويل - الحد الأقصى {MAX_CHARS} حرف للتجربة")
    voices = list_voices()
    if voice_name not in voices:
        raise gr.Error("اختار صوت من القايمة")

    progress(0.1, desc="بيتشكّل النص...")
    # only vocalize if the text is mostly unvocalized
    if len(re.findall(r"[\u064B-\u0652]", text)) < len(text) // 20:
        text = auto_tashkeel(text)

    progress(0.3, desc="بيتولّد الصوت... (أول مرة بتاخد دقايق بسبب تحميل الموديل)")
    tts = get_tts()
    v = voices[voice_name]
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    seed_val = None if seed is None or int(seed) < 0 else int(seed)
    tts.infer(ref_file=v["wav"], ref_text=v["txt"], gen_text=text,
              file_wave=out, seed=seed_val, speed=float(speed),
              nfe_step=16, cross_fade_duration=0.15,
              normalize_numbers=False, force_tashkeel=False)

    import soundfile as sf
    d, sr = sf.read(out)
    n = int(0.015 * sr)
    d[:n] = d[:n] * (1.0 * __import__("numpy").linspace(0, 1, n))
    peak = abs(d).max()
    if peak > 0.95:
        d = d * (0.95 / peak)
    sf.write(out, d, sr)
    progress(1.0, desc="خلص")
    return out


CSS = """
.gradio-container { direction: rtl; text-align: right; max-width: 820px !important; margin: auto; }
textarea, input { direction: rtl !important; text-align: right !important; }
footer { display: none !important; }
#gen-btn { font-size: 1.2em; }
"""

DESCRIPTION = """
# 🎙️ استوديو الصوت العربي — تجربة مجانية

حوّل أي نص عربي لصوت واضح بالتشكيل. اكتب النص، اختار الصوت، ودوس **ولّد الصوت**.

- مجاني بالكامل ومفتوح المصدر — بيشتغل بموديل SILMA العربي على معالج مجاني، فالنصوص الطويلة بتاخد وقت (دقيقة أو اتنين للجملة الطويلة).
- التشكيل بيتضاف تلقائي لو النص من غير تشكيل.
- أصوات الأطفال مستنسخة من تسجيلات حقيقية مفتوحة الرخصة (SpeechOcean762, CC BY 4.0).
"""


def ui():
    voices = list_voices()
    names = list(voices.keys())
    with gr.Blocks(title="استوديو الصوت العربي") as demo:
        gr.Markdown(DESCRIPTION)
        text = gr.Textbox(
            lines=6,
            label="النص",
            placeholder="اكتب أو الصق النص العربي هنا... مثال: لماذا تبدو السماء زرقاء؟",
        )
        voice = gr.Dropdown(
            choices=[(label_for(n), n) for n in names],
            value=(names[0] if names else None),
            label="الصوت",
        )
        gen = gr.Button("🎧 ولّد الصوت", variant="primary", elem_id="gen-btn")
        audio = gr.Audio(label="النتيجة — اسمع أو حمّل", type="filepath")

        with gr.Accordion("خيارات متقدمة", open=False):
            speed = gr.Slider(0.6, 1.6, value=1.0, step=0.05, label="السرعة")
            seed = gr.Number(value=-1, precision=0,
                             label="البذرة — -1 = عشوائي، رقم ثابت = نفس النتيجة")
            gr.Markdown("لتثبيت التشكيل بنفسك: شكّل النص يدويًا قبل اللصق، والأداة هتحترم تشكيلك.")

        gen.click(generate, inputs=[text, voice, speed, seed], outputs=audio)

        gr.Markdown(
            "صنع بـ [SILMA TTS](https://github.com/silma-ai/silma-tts) (Apache-2.0) "
            "ومقاطع [SpeechOcean762](https://www.openslr.org/101/) (CC BY 4.0). "
            "المشروع الكامل لسطح المكتب: [arabic-voice-studio](https://github.com/bahyfree1/arabic-voice-studio)."
        )
    return demo


if __name__ == "__main__":
    ui().launch(css=CSS, share=True)
