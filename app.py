# -*- coding: utf-8 -*-
"""Arabic Voice Studio: paste Arabic text, review the tashkeel, pick a voice, get audio.

Run:  python app.py   (then open http://127.0.0.1:7860)
"""

from pathlib import Path
import gradio as gr

import engines
from tashkeel_utils import vocalize

VOICES_DIR = Path(__file__).resolve().parent / "voices"


def list_voices():
    """Each voice is voices/<name>.wav plus voices/<name>.txt (its transcript)."""
    voices = {}
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            voices[wav.stem] = {
                "name": wav.stem,
                "wav": str(wav),
                "txt": txt.read_text(encoding="utf-8").strip(),
            }
    return voices


def ui_voices():
    return list(list_voices().keys())


def do_tashkeel(text):
    if not text.strip():
        return ""
    return vocalize(text)


ICE_GENDERS = {"ولد": "male", "بنت": "female"}
ICE_AGES = {"طفل (٦-٩ سنين)": "child", "مراهق (١٣-١٥ سنة)": "teenager",
            "شاب": "young adult", "كبير": "middle-aged"}


def do_synthesize(engine_key, voice_name, text, dialect, speed, seed,
                  ice_gender, ice_age, profile, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("اكتب النص الأول")
    progress(0.1, desc="بيتم التوليد...")
    seed_val = None if seed is None or int(seed) < 0 else int(seed)
    if engine_key in engines.TEXT_ONLY_ENGINES:
        instruct = engines.build_ice_instruct(ICE_GENDERS[ice_gender], ICE_AGES[ice_age])
        out = engines.synthesize(engine_key, None, text, dialect, speed, instruct=instruct)
    else:
        voices = list_voices()
        if not voices:
            raise gr.Error("مفيش أصوات في مجلد voices — ضيف ملف wav وملف txt بنفس الاسم")
        if voice_name not in voices:
            raise gr.Error(f"الصوت '{voice_name}' مش موجود، حدّث القايمة")
        out = engines.synthesize(engine_key, voices[voice_name], text, dialect, speed, seed=seed_val)
        import voice_profiles
        voice_profiles.apply_profile(out, profile)
    progress(1.0, desc="خلص")
    return out


CSS = """
.gradio-container { direction: rtl; text-align: right; }
textarea, input { direction: rtl !important; text-align: right !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="استوديو الصوت العربي") as demo:
    gr.Markdown("# استوديو الصوت العربي\nاكتب نص، راجع التشكيل، اختار الصوت، وحمّل الملف.")

    with gr.Row():
        engine = gr.Dropdown(
            choices=[(label, key) for key, (label, _) in engines.ENGINES.items()],
            value="habibi", label="المحرك")
        voice = gr.Dropdown(choices=ui_voices(),
                            value=(ui_voices()[0] if ui_voices() else None),
                            label="الصوت")
        refresh = gr.Button("حدّث الأصوات", scale=0)

    with gr.Row():
        dialect = gr.Dropdown(choices=["MSA", "EGY", "UNK"], value="MSA",
                              label="اللهجة (Habibi)")
        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="السرعة")
        seed = gr.Number(value=-1, precision=0,
                         label="البذرة (SILMA) — -1 = عشوائي كل مرة، رقم ثابت = نفس النتيجة")

    with gr.Row():
        ice_gender = gr.Dropdown(choices=list(ICE_GENDERS.keys()), value="بنت",
                                 label="النوع (ICE-012)")
        ice_age = gr.Dropdown(choices=list(ICE_AGES.keys()), value="طفل (٦-٩ سنين)",
                              label="العمر (ICE-012)")
        import voice_profiles
        profile = gr.Dropdown(choices=list(voice_profiles.PROFILES.keys()),
                              value="أساسي (من غير تحويل)",
                              label="الشخصية/العمر (تجريبي — تحويل صوتي، الجودة أقل من الاستنساخ)")

    raw = gr.Textbox(lines=6, label="النص الخام",
                     placeholder="اكتب أو الصق نص الدرس هنا...")
    tashkeel_btn = gr.Button("١. شكّل النص", variant="secondary")
    reviewed = gr.Textbox(lines=6, label="٢. راجع التشكيل وعدّل لو لزم (مهم للدروس)")
    gen_btn = gr.Button("٣. ولّد الصوت", variant="primary")
    audio = gr.Audio(label="النتيجة", type="filepath")

    refresh.click(lambda: gr.update(choices=ui_voices()), outputs=voice)
    tashkeel_btn.click(do_tashkeel, inputs=raw, outputs=reviewed)
    gen_btn.click(do_synthesize,
                  inputs=[engine, voice, reviewed, dialect, speed, seed, ice_gender, ice_age, profile],
                  outputs=audio)

    gr.Markdown(
        "محرك ICE-012 بيديك صوت طفل أو مراهق من النص مباشرة بدون أي تسجيل: "
        "اختار المحرك والنوع والعمر ودوس «ولّد الصوت». "
        "لإضافة صوت مستنسخ (أقرب للحقيقة): سجّل ١٠-١٥ ثانية واضحة وهادئة، "
        "واحفظها في مجلد voices باسم مثل child-8-girl.wav، ومعاها ملف "
        "child-8-girl.txt فيه نص التسجيل بالظبط، وبعدين اضغط «حدّث الأصوات».")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CSS)
