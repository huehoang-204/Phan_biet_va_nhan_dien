#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diarize_and_classify_min.py
Mục tiêu: chỉ xuất các file WAV đã gán nhãn (được nhận diện), không tạo CSV hay file phụ.

Pipeline:
  1) Diarization (NeMo: VAD MarbleNet + embedding TitaNet)
  2) Cắt WAV theo speaker (ghép liền có/không khoảng lặng, hoặc full-length/segments)
  3) Phân loại bằng .nemo fine-tune (multi-class) -> đổi tên file thành nhãn
  4) Dọn rác: xoá toàn bộ file/trung gian, chỉ chừa thư mục WAV đã gán nhãn
"""

import os, sys, json, glob, shutil
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf, ListConfig

# ==================== CẤU HÌNH ====================
INPUT_WAV       = "/hdd3/quocvm/do_an/part_002.wav"
OUT_DIR         = "/hdd3/quocvm/do_an/labeled_outputs"
DEVICE          = "cuda:0"
PUBLIC_OUT_DIR  = OUT_DIR  # ĐỂ WEB TRUY CẬP

YAML_PATH       = "/hdd3/quocvm/do_an/diar_infer_meeting.yaml"
NUM_SPEAKERS    = 4

# MODE: "concat" | "segments" | "full"
MODE            = "concat"
CONCAT_GAP_SEC  = 0.2
NORMALIZE       = True

NEMO_PATH       = "/hdd3/quocvm/do_an/titanet_ft_last.nemo"
LABELS_OVERRIDE = ["Bich", "Chien", "Chinh", "Nam"]

# === GIẢM VRAM KHI MODE == "full" ===
CLASSIFY_VIA_SPEECH_ONLY_FOR_FULL = True   # dùng speech-only để nhận diện (khuyên bật)
SPEECH_ONLY_MAX_SECONDS = 60               # tối đa X giây speech/speaker để classify (0 = không giới hạn)
# ==================================================
import matplotlib.pyplot as plt
import librosa
from matplotlib.lines import Line2D

# --- đặt cạnh các import & helpers khác ---
import re
import hashlib

# ==== PALETTE MÀU CHUẨN CHO NHÃN ====
PALETTE = {
    "Bich":  "#E53935",  # đỏ
    "Chien": "#1E88E5",  # xanh dương
    "Chinh": "#43A047",  # xanh lá
    "Nam":   "#FDD835",  # vàng
}
# key chuẩn hoá để tra nhanh
PALETTE_NORM = {"".join(c if (c.isalnum() or c in "-_.") else "_" for c in k).lower(): v
                for k, v in PALETTE.items()}

# dùng chính PALETTE_NORM như "base palette chuẩn hoá"
BASE_PALETTE = PALETTE_NORM

# fallback màu nếu nhãn không nằm trong bảng cố định
FALLBACKS = [
    "#8E24AA", "#FB8C00", "#00ACC1", "#D81B60",
    "#7CB342", "#5E35B1", "#0097A7", "#F4511E",
]

def _norm_key(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s)).lower().strip()

def _base_label(label: str) -> str:
    """Bóc nhãn 'thực' để tra màu: bỏ prefix 'speaker_', bỏ hậu tố _<số>."""
    s = str(label or "").strip()
    s = re.sub(r"^speaker_", "", s, flags=re.IGNORECASE)  # bỏ 'speaker_'
    s = re.sub(r"_\d+$", "", s)                           # bỏ hậu tố _2, _3...
    return s

def color_for(label: str) -> str:
    """
    Trả về màu cho nhãn:
      1) ưu tiên bảng màu cố định (không phân biệt hoa/thường)
      2) nếu không có, sinh màu ổn định theo hash để mỗi nhãn khác nhau có màu khác nhau
    """
    base = _base_label(label)
    k = _norm_key(base)
    if k in BASE_PALETTE:
        return BASE_PALETTE[k]

    h = int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16)
    return FALLBACKS[h % len(FALLBACKS)]

# ---------- VẼ WAVEFORM 1 MÀU (dùng cho input & từng speaker) ----------
def _load_and_norm(audio_path: str):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y.size == 0:
        raise ValueError("Audio rỗng.")
    peak = float(np.max(np.abs(y)))
    if peak > 1e-7:
        y = 0.98 * y / peak
    return y, sr

def plot_waveform_simple(
    audio_path: str,
    output_png_path: str,
    title: str = "Waveform",
    color: str = "#1c4dd9",
    downsample: Optional[int] = 2000,
):
    y, sr = _load_and_norm(audio_path)
    t = np.linspace(0.0, len(y)/sr, num=len(y), endpoint=False)
    if downsample and downsample > 0:
        step = max(1, int(sr / downsample))
        y = y[::step]
        t = t[::step]

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.2))
    ax.set_title(title, fontsize=16, pad=12, fontweight='bold')
    ax.plot(t, y, linewidth=1.8, color=color, alpha=1.0)
    ax.set_xlim(0, t[-1] if len(t) else 0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def safe_fname(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)

import re


import re

def extract_label_from_filename(path: str, base: str = None) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # Ưu tiên: <base>_speaker_<Label>(_<n>)?
    m = re.search(r'_speaker_([^_]+?)(?:_\d+)?$', stem)
    if m:
        return m.group(1)

    # Fallback: <base>_<Label>(_<n>)?
    if base and stem.startswith(base + "_"):
        stem = stem[len(base) + 1:]
    stem = re.sub(r'_\d+$', '', stem)

    return stem


def normalize_label_for_color(name: str) -> str:
    """Chuẩn hóa nhãn để tra màu: bỏ hậu tố _<số>, lowercase + safe."""
    base = re.sub(r'_\d+$', '', str(name).strip())
    return _norm_key(base)



# ---------- VẼ WAVEFORM OVERLAY THEO SPEAKER (để xem tổng thể) ----------
def plot_waveform_with_regions(
    audio_path: str,
    diarization: List[Dict],
    output_png_path: str,
    title: str = "Waveform",
    speaker_colors: Optional[Dict[str, str]] = None,
    downsample: int = 2_000,
    show_legend: bool = True,
    highlight_spk: Optional[str] = None, 
    # kiểm soát độ đậm/nhạt
    bg_alpha: float = 0.18,      # nền xám (toàn bộ waveform)
    alpha_high: float = 1.00,    # đoạn của speaker được highlight
    alpha_low: float  = 0.20,    # các speaker khác (để mờ)
    lw_high: float = 1.8,        # linewidth của highlight
    lw_low: float  = 1.0,    
):
    """
    Nền xám nhạt (toàn bộ sóng) + overlay màu cho từng đoạn có speaker.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y.size == 0:
        raise ValueError("Audio rỗng.")
    peak = float(np.max(np.abs(y)))
    if peak > 1e-7:
        y = 0.98 * y / peak

    t = np.linspace(0.0, len(y) / sr, num=len(y), endpoint=False)

    spk_list = sorted(set(d.get('spk', 'unknown') for d in diarization))

    # helper tra màu: thử nhãn gốc -> key chuẩn -> mặc định
    def _get_color(name: str) -> str:
        if speaker_colors:
            base_k = normalize_label_for_color(name)
            return (speaker_colors.get(name)
                    or speaker_colors.get(_norm_key(name))
                    or speaker_colors.get(base_k)
                    or PALETTE.get(name)
                    or PALETTE_NORM.get(_norm_key(name))
                    or PALETTE_NORM.get(base_k)
                    or "#444")
        return (PALETTE.get(name)
                or PALETTE_NORM.get(_norm_key(name))
                or PALETTE_NORM.get(normalize_label_for_color(name))
                or "#444")


    segs = []
    for d in diarization:
        s = float(d.get("start", 0.0))
        e = float(d.get("end", s + float(d.get("dur", 0.0))))
        if e > s:
            segs.append((max(0.0, s), min(len(y)/sr, e), str(d.get("spk", "unknown"))))
    segs.sort(key=lambda x: x[0])

    if downsample and downsample > 0:
        step = max(1, int(sr / downsample))
        y_bg = y[::step]; t_bg = t[::step]
    else:
        step = 1
        y_bg, t_bg = y, t

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.2))
    ax.set_title(title, fontsize=16, pad=12, fontweight='bold')
    ax.plot(t_bg, y_bg, linewidth=0.7, color="#c9d1de", alpha=bg_alpha)

    for (s, e, spk) in segs:
        i0 = max(0, int(round(s * sr)))
        i1 = min(len(y), int(round(e * sr)))
        if i1 <= i0:
            continue

        if step > 1:
            j0 = i0 // step
            j1 = max(j0 + 1, i1 // step)
            yt = y_bg[j0:j1]; tt = t_bg[j0:j1]
        else:
            yt = y[i0:i1]; tt = t[i0:i1]
            
        is_high = (highlight_spk is None) or (spk == highlight_spk)
        ax.plot(
            tt, yt,
            linewidth=(lw_high if is_high else lw_low),
            color=color_for(spk),          # <<< dùng color_for
            alpha=(alpha_high if is_high else alpha_low),
        )

    ax.set_xlim(0, len(y)/sr)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")

    if show_legend and len(spk_list) <= 12:
        handles = [Line2D([0], [0], color=color_for(s), lw=2, label=_base_label(s)) for s in spk_list]
        leg = ax.legend(handles=handles, ncols=min(4, len(handles)),
                        loc="upper right", frameon=True, fontsize=10)
        leg.get_frame().set_alpha(0.9)


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------- TIỆN ÍCH ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_wav_mono_float32(path: str):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x[:, 0]
    return x, int(sr)

def norm_peak(a: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    return a if peak <= 1e-7 else (0.99 * a / peak)

# ---------- MANIFEST / RTTM ----------
def write_manifest(audio_path: str, manifest_path: str, num_spk: Optional[int]):
    rec = {
        "audio_filepath": os.path.abspath(audio_path),
        "offset": 0, "duration": None,
        "label": "infer", "text": "-",
        "num_speakers": int(num_spk) if num_spk else None,
        "rttm_filepath": None, "uem_filepath": None,
    }
    ensure_dir(os.path.dirname(manifest_path) or ".")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def parse_rttm(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3]); dur = float(parts[4])
            except Exception:
                continue
            name = parts[7] if len(parts) >= 8 else "speaker_0"
            out.append({"start": start, "dur": dur, "spk": name})
    out.sort(key=lambda d: d["start"])
    return out

def find_rttm_for_audio(out_dir: str, audio_path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cands = [
        os.path.join(out_dir, "pred_rttms", f"{base}.rttm"),
        os.path.join(out_dir, f"{base}.rttm"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    pr = os.path.join(out_dir, "pred_rttms")
    if os.path.isdir(pr):
        for p in sorted(glob.glob(os.path.join(pr, "*.rttm"))):
            if os.path.basename(p).startswith(base):
                return p
    return None

# ---------- EXPORT WAV ----------
def export_segments(x, sr, entries, base, outdir, normalize=True, fallback_unknown="Unknown"):
    ensure_dir(outdir)
    for e in entries:
        s = int(round(e["start"] * sr))
        e_idx = int(round((e["start"] + e["dur"]) * sr))
        seg = norm_peak(x[s:e_idx]) if normalize else x[s:e_idx]
        s_ms = int(round(1000 * e["start"]))
        e_ms = int(round(1000 * (e["start"] + e["dur"])))
        spk = (e.get("spk") or "").strip() or fallback_unknown
        fname = f"{base}_{s_ms:08d}-{e_ms:08d}_{safe_fname(spk)}.wav"
        sf.write(os.path.join(outdir, fname), seg, sr, subtype="PCM_16")

def export_concat(x, sr, entries, base, outdir, gap_sec=0.2, normalize=True, fallback_unknown="Unknown"):
    ensure_dir(outdir)
    by_spk: Dict[str, List[np.ndarray]] = {}
    for e in entries:
        s = int(round(e["start"] * sr))
        e_idx = int(round((e["start"] + e["dur"]) * sr))
        spk = (e.get("spk") or "").strip() or fallback_unknown
        by_spk.setdefault(spk, []).append(x[s:e_idx])
    gap = np.zeros(int(round(max(0.0, gap_sec) * sr)), dtype=np.float32) if gap_sec > 0 else None
    for spk, arrs in by_spk.items():
        if not arrs:
            continue
        if gap is not None and len(arrs) > 1:
            parts = []
            for i, seg in enumerate(arrs):
                parts.append(seg)
                if i != len(arrs) - 1:
                    parts.append(gap)
            cat = np.concatenate(parts, axis=0)
        else:
            cat = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
        cat = norm_peak(cat) if normalize else cat
        out_path = os.path.join(outdir, f"{base}_{safe_fname(spk)}.wav")
        sf.write(out_path, cat, sr, subtype="PCM_16")

def export_full_length(
    x: np.ndarray,
    sr: int,
    entries: List[Dict],
    base: str,
    outdir: str,
    normalize: bool = True,
    fallback_unknown: str = "Unknown",
):
    """
    Mỗi speaker -> 1 file dài bằng audio gốc.
    Chỗ không nói = 0. Chỗ nói = chép mẫu gốc vào đúng vị trí thời gian.
    """
    ensure_dir(outdir)
    duration = len(x)
    by_spk: Dict[str, np.ndarray] = {}

    for e in entries:
        s = int(round(e["start"] * sr))
        e_idx = int(round((e["start"] + e["dur"]) * sr))
        s = max(0, min(s, duration))
        e_idx = max(0, min(e_idx, duration))
        if e_idx <= s:
            continue
        spk = (e.get("spk") or "").strip() or fallback_unknown
        if spk not in by_spk:
            by_spk[spk] = np.zeros_like(x, dtype=np.float32)
        by_spk[spk][s:e_idx] = x[s:e_idx]

    for spk, y in by_spk.items():
        y = norm_peak(y) if normalize else y
        out_path = os.path.join(outdir, f"{base}_{safe_fname(spk)}.wav")
        sf.write(out_path, y, sr, subtype="PCM_16")

def export_concat_for_classify(
    x, sr, entries, base, outdir, max_seconds=0, normalize=True, fallback_unknown="Unknown"
):
    ensure_dir(outdir)
    by_spk: Dict[str, List[np.ndarray]] = {}
    for e in entries:
        s = int(round(e["start"] * sr))
        e_idx = int(round((e["start"] + e["dur"]) * sr))
        spk = (e.get("spk") or "").strip() or fallback_unknown
        by_spk.setdefault(spk, []).append(x[s:e_idx])

    for spk, arrs in by_spk.items():
        if not arrs:
            continue
        cat = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
        if max_seconds and max_seconds > 0:
            max_len = int(max_seconds * sr)
            cat = cat[:max_len]
        y = norm_peak(cat) if normalize else cat
        out_path = os.path.join(outdir, f"{base}_{safe_fname(spk)}.wav")
        sf.write(out_path, y, sr, subtype="PCM_16")

# ---------- NeMo Diarizer ----------
def load_diarizer_class():
    import importlib
    for mod, cls in [
        ("nemo.collections.asr.models.clustering_diarizer", "ClusteringDiarizer"),
        ("nemo.collections.asr.models.msdd_diarizer", "MSDDDiarizer"),
    ]:
        try:
            m = importlib.import_module(mod)
            C = getattr(m, cls, None)
            if C and hasattr(C, "diarize"):
                return C
        except Exception:
            pass
    return None

def _as_list_of_floats(x, default):
    if isinstance(x, ListConfig) or isinstance(x, list):
        return [float(v) for v in x]
    if x is None:
        return [float(v) for v in default]
    return [float(x)]

def apply_sane_overrides(cfg, manifest_path, out_dir):
    cfg.device = DEVICE
    if "diarizer" not in cfg:
        cfg.diarizer = OmegaConf.create()
    cfg.diarizer.manifest_filepath = os.path.abspath(manifest_path)
    cfg.diarizer.out_dir = os.path.abspath(out_dir)

    if "vad" not in cfg.diarizer:
        cfg.diarizer.vad = OmegaConf.create()
    vad = cfg.diarizer.vad
    if "parameters" not in vad:
        vad.parameters = OmegaConf.create()
    vp = vad.parameters

    vp.window_length_in_sec = float(vp.get("window_length_in_sec", 0.15))
    vp.shift_length_in_sec  = float(vp.get("shift_length_in_sec", 0.01))
    vp.onset  = float(vp.get("onset", 0.5))
    vp.offset = float(vp.get("offset", 0.5))
    vp.min_duration_on  = float(vp.get("min_duration_on", 0.2))
    vp.min_duration_off = float(vp.get("min_duration_off", 0.1))
    vp.smoothing = vp.get("smoothing", True)
    vp.smoothing_method = str(vp.get("smoothing_method", "median"))
    vp.smoothing_size   = int(vp.get("smoothing_size", 7))
    vp.overlap = float(vp.get("overlap", 0.5))
    vp.generate_overlap = bool(vp.get("generate_overlap", True))
    vp.use_gpu = bool(vp.get("use_gpu", True))

    if "speaker_embeddings" not in cfg.diarizer:
        cfg.diarizer.speaker_embeddings = OmegaConf.create()
    se = cfg.diarizer.speaker_embeddings
    se.model_path = "titanet_large"
    if "parameters" not in se:
        se.parameters = OmegaConf.create()
    sp = se.parameters

    win_default   = [1.5, 1.0, 0.5]
    shift_default = [0.75, 0.5, 0.25]
    win_list   = _as_list_of_floats(sp.get("window_length_in_sec", win_default), win_default)
    shift_list = _as_list_of_floats(sp.get("shift_length_in_sec",  shift_default), shift_default)
    L = min(len(win_list), len(shift_list))
    if L == 0:
        win_list, shift_list = win_default, shift_default
        L = len(win_list)
    else:
        win_list   = win_list[:L]
        shift_list = shift_list[:L]
    weights = sp.get("multiscale_weights", [1.0] * L)
    if isinstance(weights, (int, float)):
        weights = [float(weights)] * L
    else:
        weights = _as_list_of_floats(weights, [1.0] * L)
        if len(weights) != L:
            weights = [1.0] * L
    sp.window_length_in_sec = win_list
    sp.shift_length_in_sec  = shift_list
    sp.multiscale_weights   = weights

    return cfg

# ---------- LẤY NHÃN ----------
def try_load_labels_from_cfg(cfg):
    labs = getattr(cfg, "labels", None)
    if isinstance(labs, (list, tuple)) and len(labs) > 0:
        return list(labs)
    for k in ("train_ds", "validation_ds"):
        ds = getattr(cfg, k, None)
        if ds and hasattr(ds, "labels"):
            labs2 = getattr(ds, "labels")
            if isinstance(labs2, (list, tuple)) and len(labs2) > 0:
                return list(labs2)
    return None

def try_load_labels_sidecar(nemo_path):
    path = os.path.join(os.path.dirname(os.path.abspath(nemo_path)), "labels.txt")
    if os.path.isfile(path):
        labs = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                if "," in ln:
                    labs += [p.strip() for p in ln.split(",") if p.strip()]
                else:
                    labs.append(ln)
        if labs:
            return labs
    return None

# ---------- PHÂN LOẠI ----------
def classify_and_rename_inplace(nemo_path: str, folder: str, labels_override: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Chạy classify tất cả wav trong 'folder', đổi tên theo nhãn dự đoán và
    trả ra mapping {<old_token>: <label>}, với old_token là phần cuối tên file
    (vd 'speaker_0' trong 'base_speaker_0.wav').
    """
    mapping: Dict[str, str] = {}
    if not os.path.isfile(nemo_path):
        return mapping

    import torch, torchaudio
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncDecSpeakerLabelModel.restore_from(nemo_path, map_location=device, strict=False).eval().to(device)

    # lấy nhãn
    labels = None
    if labels_override: labels = list(labels_override)
    if labels is None: labels = try_load_labels_sidecar(nemo_path)
    if labels is None: labels = try_load_labels_from_cfg(model.cfg)
    if labels is None:
        n = int(getattr(getattr(model.cfg, "decoder", None), "num_classes", 0) or 0)
        labels = [str(i) for i in range(n)] if n > 0 else []

    target_sr = int(getattr(getattr(model.cfg, "preprocessor", None), "sample_rate", 16000))

    def forward_logits(x, xlen):
        try:
            out = model(input_signal=x, input_signal_length=xlen)
        except TypeError:
            out = model(audio_signal=x, length=xlen)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            return out["logits"]
        return out

    wavs = sorted(glob.glob(os.path.join(folder, "*.wav")))
    import torch
    for wp in wavs:
        base = os.path.basename(wp)
        wav, sr = sf.read(wp, dtype="float32", always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = wav[:, 0]
        if sr != target_sr:
            t = torch.from_numpy(wav).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            wav = t.squeeze(0).numpy()
            sr = target_sr

        X = torch.from_numpy(wav).unsqueeze(0).to(device)
        xlen = torch.tensor([X.shape[-1]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = forward_logits(X, xlen)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        
        idx = int(np.argmax(probs))

        # Nhãn ưu tiên:
        #  - LABELS_OVERRIDE nếu có (chuẩn nhất)
        #  - labels.txt cạnh .nemo nếu có
        #  - labels trong model.cfg
        # => Nếu vẫn không có nhãn chữ hợp lệ thì raise để không bao giờ rơi về số.
        if not labels or idx >= len(labels):
            raise RuntimeError(
                f"Classifier không có danh sách nhãn hợp lệ (labels) hoặc idx={idx} vượt biên. "
                "Hãy cung cấp LABELS_OVERRIDE hoặc labels.txt cạnh file .nemo."
            )

        pred_label = str(labels[idx]).strip()
        if not pred_label or pred_label.isdigit():
            raise RuntimeError(
                f"Classifier trả nhãn không hợp lệ: '{pred_label}'. "
                "Hãy kiểm tra LABELS_OVERRIDE / labels.txt."
            )

        
        stem = os.path.splitext(base)[0]
        m = re.search(r'(speaker_\d+)(?:_\d+)?$', stem)
        if m:
            old_token = m.group(1)          # ví dụ 'speaker_0'
        else:
            # fallback: vẫn lấy token cuối nếu không có chuỗi 'speaker_'
            old_token = stem.split("_")[-1]

        mapping[old_token] = pred_label  # CHUẨN: không thêm "speaker_" nữa
        # Bổ sung key tương đương để khớp được với RTTM
        if old_token.startswith("speaker_"):
            num = old_token.split("_")[-1]
            if num.isdigit():
                mapping[num] = pred_label
        elif old_token.isdigit():
            mapping[f"speaker_{old_token}"] = pred_label

        # đổi tên file ngay trong 'folder'
        root_no_last = "_".join(os.path.splitext(base)[0].split("_")[:-1]) or os.path.splitext(base)[0]
        new_name = f"{root_no_last}_{safe_fname(pred_label)}.wav"
        new_path = os.path.join(os.path.dirname(wp), new_name)
        if os.path.abspath(new_path) != os.path.abspath(wp):
            if os.path.exists(new_path):
                i = 2
                stem = os.path.splitext(new_name)[0]
                while True:
                    cand = os.path.join(os.path.dirname(wp), f"{stem}_{i}.wav")
                    if not os.path.exists(cand):
                        new_path = cand; break
                    i += 1
            os.replace(wp, new_path)
    return mapping

# ---------- RUN PIPELINE ----------
def run_pipeline(
    input_wav: str,
    num_speakers: Optional[int] = None,
    mode: Optional[str] = None,
    concat_gap_sec: Optional[float] = None,
    normalize: Optional[bool] = None,
) -> Dict[str, object]:
    if not os.path.isfile(input_wav):
        raise FileNotFoundError(f"Không thấy file audio: {input_wav}")
    if not os.path.isfile(YAML_PATH):
        raise FileNotFoundError(f"Không thấy YAML_PATH: {YAML_PATH}")

    base = os.path.splitext(os.path.basename(input_wav))[0]
    tmp_dir   = os.path.join(OUT_DIR, "__tmp_diar_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    final_dir = os.path.join(OUT_DIR, safe_fname(base))
    ensure_dir(tmp_dir); ensure_dir(final_dir)

    # manifest
    manifest = os.path.join(tmp_dir, "manifest.json")
    write_manifest(input_wav, manifest, num_speakers if num_speakers is not None else NUM_SPEAKERS)

    # diarize
    cfg = OmegaConf.load(YAML_PATH)
    cfg = apply_sane_overrides(cfg, manifest, tmp_dir)
    Diarizer = load_diarizer_class()
    if Diarizer is None:
        raise RuntimeError("Không tìm thấy diarizer class.")
    try:
        diarizer = Diarizer(cfg=cfg)
    except TypeError:
        diarizer = Diarizer(config=cfg)
    diarizer.diarize()

    # parse RTTM
    rttm = find_rttm_for_audio(tmp_dir, input_wav)
    if not rttm:
        raise RuntimeError("Không tìm thấy RTTM.")
    entries = parse_rttm(rttm)
    if not entries:
        raise RuntimeError("RTTM rỗng hoặc sai định dạng.")

    # THÊM end = start + dur
    for e in entries:
        e["end"] = e["start"] + e["dur"]

    x, sr = read_wav_mono_float32(input_wav)
    effective_mode = (mode or MODE).lower()
    effective_norm = normalize if normalize is not None else NORMALIZE
    effective_gap  = float(concat_gap_sec if concat_gap_sec is not None else CONCAT_GAP_SEC)

    # --- XUẤT FILE THEO MODE ---
    if effective_mode == "segments":
        export_segments(x, sr, entries, base, final_dir, normalize=effective_norm)
        mapping = classify_and_rename_inplace(NEMO_PATH, final_dir, labels_override=LABELS_OVERRIDE)

    elif effective_mode == "full":
        export_full_length(x, sr, entries, base, final_dir, normalize=effective_norm)
        if CLASSIFY_VIA_SPEECH_ONLY_FOR_FULL:
            tmp_cls_dir = os.path.join(final_dir, "__cls_tmp")
            ensure_dir(tmp_cls_dir)
            export_concat_for_classify(
                x, sr, entries, base, tmp_cls_dir,
                max_seconds=SPEECH_ONLY_MAX_SECONDS,
                normalize=True
            )
            mapping = classify_and_rename_inplace(NEMO_PATH, tmp_cls_dir, labels_override=LABELS_OVERRIDE)
            for e in entries:
                e["spk"] = mapping.get(e["spk"], e["spk"])

            # đổi tên các file FULL trong final_dir theo mapping
            full_files = glob.glob(os.path.join(final_dir, "*.wav"))
            for p in full_files:
                bn = os.path.basename(p)
                tok = os.path.splitext(bn)[0].split("_")[-1]   # ví dụ: speaker_0
                new_lbl = mapping.get(tok, tok)
                newp = os.path.join(final_dir, bn.replace(tok + ".wav", safe_fname(new_lbl) + ".wav"))
                if newp != p:
                    if os.path.exists(newp):
                        stem = os.path.splitext(newp)[0]; i = 2
                        while True:
                            cand = f"{stem}_{i}.wav"
                            if not os.path.exists(cand):
                                newp = cand; break
                            i += 1
                    os.replace(p, newp)

            shutil.rmtree(tmp_cls_dir, ignore_errors=True)
        else:
            mapping = classify_and_rename_inplace(NEMO_PATH, final_dir, labels_override=LABELS_OVERRIDE)
            for e in entries:
                e["spk"] = mapping.get(e["spk"], e["spk"])

    else:
        export_concat(x, sr, entries, base, final_dir, gap_sec=effective_gap, normalize=effective_norm)
        mapping = classify_and_rename_inplace(NEMO_PATH, final_dir, labels_override=LABELS_OVERRIDE)

    # đồng bộ entries (nếu chưa làm ở trên)
    if effective_mode in ("segments", "concat"):
        for e in entries:
            e["spk"] = mapping.get(e["spk"], e["spk"])

    # --- DỌN TMP ---
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # --- KẾT QUẢ ---
    files = sorted(glob.glob(os.path.join(final_dir, "*.wav")))
    result = {"final_dir": final_dir, "files": files}
    result["diarization"] = [{"start": e["start"], "end": e["end"], "spk": e["spk"]} for e in entries]
    spk_list = sorted(set(e["spk"] for e in entries))  # <-- BẮT BUỘC CÓ

    # BẢNG MÀU CHUẨN
    base_palette = {"Bich":"#E53935","Chien":"#1E88E5","Chinh":"#43A047","Nam":"#FDD835"}
    fallback = ["#8E24AA", "#FB8C00", "#00ACC1", "#D81B60"]
    base_palette_norm = {_norm_key(k): v for k, v in base_palette.items()}

    spk_to_color: Dict[str, str] = {}
    for i, spk in enumerate(spk_list):
        col = base_palette_norm.get(_norm_key(spk), fallback[i % len(fallback)])
        # map gốc
        spk_to_color[spk] = col
        # map chuẩn hóa
        spk_to_color[_norm_key(spk)] = col
        # map base-name (phòng trường hợp *_<số>)
        spk_to_color[normalize_label_for_color(spk)] = col

    # === 1. INPUT WAVEFORM (1 màu) ===
    base_name = os.path.splitext(os.path.basename(input_wav))[0]
    input_wave_png = os.path.join(final_dir, f"{base_name}_INPUT_WAVEFORM.png")
    plot_waveform_simple(
        audio_path=input_wav,
        output_png_path=input_wave_png,
        title=f"{base_name} - Input",
        color="#1c4dd9",
    )
    result["input_waveform_url"] = os.path.relpath(input_wave_png, PUBLIC_OUT_DIR)

    # === 2. OUTPUT WAVEFORM (overlay theo speaker để xem timeline) ===
    output_wave_png = os.path.join(final_dir, f"{base_name}_OUTPUT_WAVEFORM.png")
    plot_waveform_with_regions(
        audio_path=input_wav,
        diarization=entries,
        output_png_path=output_wave_png,
        title=f"{base_name} - Output (Diarized)",
        speaker_colors=spk_to_color
    )
    result["output_waveform_url"] = os.path.relpath(output_wave_png, PUBLIC_OUT_DIR)

    # === 3. SPEAKER WAVEFORMS ===
    result["speaker_waveforms"] = {}
    base_name = os.path.splitext(os.path.basename(input_wav))[0]

    for wav_file in files:
        # luôn lấy nhãn chuẩn từ tên file (hỗ trợ cả dạng "<base>_speaker_<Label>.wav")
        label = extract_label_from_filename(wav_file, base=base_name)

        sp_png = os.path.join(final_dir, f"{label}_WAVEFORM.png")
        plot_waveform_simple(
            audio_path=wav_file,
            output_png_path=sp_png,
            title=label,
            color=color_for(label),     # <--- DÙNG color_for
            downsample=2000
        )
        result["speaker_waveforms"][label] = os.path.relpath(sp_png, PUBLIC_OUT_DIR)

    return result

# ---------- MAIN ----------
def main():
    if not os.path.isfile(INPUT_WAV):
        print(f"[ERR] Không thấy INPUT_WAV: {INPUT_WAV}"); sys.exit(2)
    if not os.path.isfile(YAML_PATH):
        print(f"[ERR] Không thấy YAML_PATH: {YAML_PATH}"); sys.exit(3)

    base = os.path.splitext(os.path.basename(INPUT_WAV))[0]
    TMP_DIR = os.path.join(OUT_DIR, "__tmp_diar_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    FINAL_DIR = os.path.join(OUT_DIR, safe_fname(base))
    ensure_dir(TMP_DIR); ensure_dir(FINAL_DIR)

    manifest = os.path.join(TMP_DIR, "manifest.json")
    write_manifest(INPUT_WAV, manifest, NUM_SPEAKERS)

    cfg = OmegaConf.load(YAML_PATH)
    cfg = apply_sane_overrides(cfg, manifest, TMP_DIR)

    Diarizer = load_diarizer_class()
    if Diarizer is None:
        print("[ERR] Không tìm thấy diarizer class."); sys.exit(5)
    try:
        diarizer = Diarizer(cfg=cfg)
    except TypeError:
        diarizer = Diarizer(config=cfg)
    diarizer.diarize()

    rttm = find_rttm_for_audio(TMP_DIR, INPUT_WAV)
    if not rttm:
        print("[ERR] Không tìm thấy RTTM."); sys.exit(6)
    entries = parse_rttm(rttm)
    if not entries:
        print("[ERR] RTTM rỗng hoặc sai định dạng."); sys.exit(7)

    for e in entries:
        e["end"] = e["start"] + e["dur"]

    x, sr = read_wav_mono_float32(INPUT_WAV)
    if MODE == "segments":
        export_segments(x, sr, entries, base, FINAL_DIR, normalize=NORMALIZE)
    elif MODE == "full":
        export_full_length(x, sr, entries, base, FINAL_DIR, normalize=NORMALIZE)
        if CLASSIFY_VIA_SPEECH_ONLY_FOR_FULL:
            tmp_cls_dir = os.path.join(FINAL_DIR, "__cls_tmp")
            ensure_dir(tmp_cls_dir)
            export_concat_for_classify(x, sr, entries, base, tmp_cls_dir, max_seconds=SPEECH_ONLY_MAX_SECONDS, normalize=True)
            mapping = classify_and_rename_inplace(NEMO_PATH, tmp_cls_dir, labels_override=LABELS_OVERRIDE)
            for e in entries:
                e["spk"] = mapping.get(e["spk"], e["spk"])
            for p in glob.glob(os.path.join(FINAL_DIR, "*.wav")):
                bn = os.path.basename(p)
                tok = os.path.splitext(bn)[0].split("_")[-1]
                new_lbl = mapping.get(tok, tok)
                newp = os.path.join(FINAL_DIR, bn.replace(tok + ".wav", safe_fname(new_lbl) + ".wav"))
                if newp != p:
                    if os.path.exists(newp):
                        stem = os.path.splitext(newp)[0]; i = 2
                        while True:
                            cand = f"{stem}_{i}.wav"
                            if not os.path.exists(cand): newp = cand; break
                            i += 1
                    os.replace(p, newp)
            shutil.rmtree(tmp_cls_dir, ignore_errors=True)
        else:
            classify_and_rename_inplace(NEMO_PATH, FINAL_DIR, labels_override=LABELS_OVERRIDE)
    else:
        export_concat(x, sr, entries, base, FINAL_DIR, gap_sec=CONCAT_GAP_SEC, normalize=NORMALIZE)
        classify_and_rename_inplace(NEMO_PATH, FINAL_DIR, labels_override=LABELS_OVERRIDE)

    try:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    except Exception:
        pass

    print(f"[DONE] WAV đã gán nhãn nằm tại: {FINAL_DIR}")

if __name__ == "__main__":
    main()
