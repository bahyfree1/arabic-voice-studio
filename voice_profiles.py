# -*- coding: utf-8 -*-
"""Voice age/gender profiles: DSP pitch+formant presets over generated speech.

These are PRESETS, not separate AI voices and not recordings: the engine
generates the base audio, then we scale pitch (f0) and the spectral envelope
(formants) with the WORLD vocoder to approximate a younger/smaller or
older/larger vocal tract. No recording is needed from the user. The result is
convincing for lessons, but it is a transformation of the base voice, so a
real child recording cloned with Habibi/SILMA still sounds more natural.

Each preset is (f0_scale, formant_scale) relative to the base voice.
"""

PROFILES = {
    "أساسي (من غير تحويل)": None,
    "بنت ٦ سنين": (1.68, 1.24),
    "بنت ٧ سنين": (1.56, 1.19),
    "بنت ٨ سنين": (1.45, 1.14),
    "ولد ٦ سنين": (1.56, 1.21),
    "ولد ٧ سنين": (1.45, 1.16),
    "ولد ٨ سنين": (1.33, 1.11),
    "بنت ١٥ سنة": (1.16, 1.06),
    "ولد ١٥ سنة": (0.86, 0.97),
    "راجل كبير": (0.60, 0.88),
}


def apply_profile(wav_path, profile_label):
    """Apply a profile preset to a WAV file in place. Returns the path."""
    scales = PROFILES.get(profile_label)
    if not scales:
        return wav_path
    f0_scale, formant_scale = scales
    import numpy as np
    import soundfile as sf
    import pyworld as pw

    d, sr = sf.read(str(wav_path))
    d = d.astype(np.float64)
    _f0, t = pw.dio(d, sr, f0_floor=80, f0_ceil=500)
    f0 = pw.stonemask(d, _f0, t, sr)
    sp = pw.cheaptrick(d, f0, t, sr)
    ap = pw.d4c(d, f0, t, sr)
    idx = np.minimum((np.arange(sp.shape[1]) * formant_scale).astype(int), sp.shape[1] - 1)
    sp2 = np.ascontiguousarray(sp[:, idx])
    y = pw.synthesize(np.ascontiguousarray(f0 * f0_scale), sp2, ap, sr)
    peak = float(np.abs(y).max())
    if peak > 0.95:
        y = y * (0.95 / peak)
    sf.write(str(wav_path), y, sr)
    return wav_path
