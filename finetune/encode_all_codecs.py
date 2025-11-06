#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import XcodecModel, AutoFeatureExtractor

# ---------- helpers ----------

def stem_kind(name: str) -> Optional[str]:
    if name.endswith(".Vocals.wav"): return "Vocals"
    if name.endswith(".Instrumental.wav"): return "Instrumental"
    return None

def base_key_from_stem(p: Path) -> str:
    # strip ".Vocals.wav" / ".Instrumental.wav"
    return re.sub(r"\.(Vocals|Instrumental)\.wav$", "", p.name)

def base_key_from_mix(p: Path) -> str:
    # strip ".wav"
    return re.sub(r"\.wav$", "", p.name)

def to_mono(wav: np.ndarray) -> np.ndarray:
    # wav shape: (T,) or (T,C) -> (T,)
    if wav.ndim == 2:
        return wav.mean(axis=1).astype(np.float32)
    return wav.astype(np.float32)

def resample_1d(wav_1d: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr: return wav_1d
    x = torch.from_numpy(wav_1d).unsqueeze(0)  # (1, T)
    y = torchaudio.transforms.Resample(sr, target_sr)(x)
    return y.squeeze(0).numpy()

def encode_1d_mono(
    wav_1d: np.ndarray,
    sr: int,
    model: XcodecModel,
    fe: AutoFeatureExtractor,
    device: str = "cuda",
    chunk_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Encode mono 1-D audio with optional chunking.
    Returns int64 codes of shape (num_quantizers, total_code_len).
    """
    target_sr = fe.sampling_rate
    wav_1d = resample_1d(wav_1d, sr, target_sr)

    def _encode_chunk(x_1d: np.ndarray) -> np.ndarray:
        inputs = fe(raw_audio=x_1d, sampling_rate=target_sr, return_tensors="pt")
        iv = inputs["input_values"]
        if iv.ndim == 1:      # (T,) -> (1,1,T)
            iv = iv.unsqueeze(0).unsqueeze(0)
        elif iv.ndim == 2:    # (1,T) -> (1,1,T)
            iv = iv.unsqueeze(1)
        iv = iv.to(device)
        with torch.no_grad():
            out = model.encode(iv)
            codes = out.audio_codes[0].cpu().numpy().astype(np.int64)  # (Q, L)
        return codes

    if not chunk_seconds:
        return _encode_chunk(wav_1d)

    # chunk with small overlap to be safe (codes are usually robust; overlap mostly harmless)
    hop = int(chunk_seconds * target_sr)
    if hop <= 0 or hop >= wav_1d.shape[0]:
        return _encode_chunk(wav_1d)

    chunks: List[np.ndarray] = []
    start = 0
    while start < wav_1d.shape[0]:
        end = min(wav_1d.shape[0], start + hop)
        chunk = wav_1d[start:end]
        codes = _encode_chunk(chunk)
        chunks.append(codes)
        start = end

    # concat along time axis (code_len)
    # ensure consistent num_quantizers
    q = chunks[0].shape[0]
    for c in chunks:
        assert c.shape[0] == q, "Inconsistent quantizer count across chunks"
    return np.concatenate(chunks, axis=1)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix_dir", default="example/wav", help="Folder with full-mix WAVs")
    ap.add_argument("--stems_dir", default="example/sep", help="Folder with stems (*.Vocals.wav / *.Instrumental.wav)")
    ap.add_argument("--out_dir", default="example/npy", help="Where to write .npy codes")
    ap.add_argument("--jsonl", default="example/jsonl/voidvoid_all.jsonl", help="Output JSONL with codec paths")
    ap.add_argument("--model_id", default="hf-audio/xcodec-hubert-librispeech")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tags", default="rock, VOIDVOID")
    ap.add_argument("--chunk_seconds", type=float, default=None, help="Optional chunk size to avoid OOM")
    args = ap.parse_args()

    mix_dir = Path(args.mix_dir).resolve()
    stems_dir = Path(args.stems_dir).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl).resolve(); jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    model = XcodecModel.from_pretrained(args.model_id).to(args.device)
    fe = AutoFeatureExtractor.from_pretrained(args.model_id)
    target_sr = fe.sampling_rate

    # index full mixes
    mixes: Dict[str, Path] = {}
    if mix_dir.is_dir():
        for p in mix_dir.rglob("*.wav"):
            mixes[base_key_from_mix(p)] = p

    # index stems
    vocals: Dict[str, Path] = {}
    inst: Dict[str, Path] = {}
    if stems_dir.is_dir():
        for p in stems_dir.rglob("*.wav"):
            kind = stem_kind(p.name)
            if kind == "Vocals":
                vocals[base_key_from_stem(p)] = p
            elif kind == "Instrumental":
                inst[base_key_from_stem(p)] = p

    # union keys
    keys = sorted(set(mixes) | set(vocals) | set(inst))
    print(f"Found {len(keys)} tracks (mixes: {len(mixes)}, vocals: {len(vocals)}, instr: {len(inst)})")

    made = 0
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, key in enumerate(keys, 1):
            mix_wav = mixes.get(key)
            v_wav = vocals.get(key)
            i_wav = inst.get(key)

            # --- determine audio length (prefer mix, else longest stem) ---
            length_sec = None
            cand_lengths = []
            for candidate in [mix_wav, v_wav, i_wav]:
                if candidate and candidate.is_file():
                    info = sf.info(str(candidate))
                    cand_lengths.append(info.frames / info.samplerate)
            if cand_lengths:
                length_sec = float(max(cand_lengths))

            # --- encode FULL MIX (codec) ---
            codec_path = None
            try:
                if mix_wav and mix_wav.is_file():
                    x, sr = sf.read(str(mix_wav), dtype="float32", always_2d=True)
                    mix_1d = to_mono(x)
                else:
                    # build mix from available stems
                    wavs_1d = []
                    srs = []
                    if v_wav and v_wav.is_file():
                        xv, srv = sf.read(str(v_wav), dtype="float32", always_2d=True)
                        wavs_1d.append(to_mono(xv)); srs.append(srv)
                    if i_wav and i_wav.is_file():
                        xi, sri = sf.read(str(i_wav), dtype="float32", always_2d=True)
                        wavs_1d.append(to_mono(xi)); srs.append(sri)
                    if not wavs_1d:
                        raise FileNotFoundError("No mix or stems available for codec")
                    # resample & align
                    wavs_rs = [resample_1d(w, sr, target_sr) for w, sr in zip(wavs_1d, srs)]
                    min_len = min(w.shape[0] for w in wavs_rs)
                    wavs_rs = [w[:min_len] for w in wavs_rs]
                    mix_1d = np.sum(np.stack(wavs_rs, axis=0), axis=0).astype(np.float32)
                    # normalize
                    peak = np.max(np.abs(mix_1d)) if mix_1d.size else 0.0
                    if peak > 0: mix_1d = 0.99 * mix_1d / peak
                    sr = target_sr

                codec_codes = encode_1d_mono(mix_1d, sr, model, fe, device=args.device, chunk_seconds=args.chunk_seconds)
                codec_path = out_dir / f"{key}.npy"
                np.save(codec_path, codec_codes)
            except Exception as e:
                print(f"[{key}] WARN codec: {e}")

            # --- encode VOCALS stem (if present) ---
            v_codec_path = None
            if v_wav and v_wav.is_file():
                try:
                    xv, srv = sf.read(str(v_wav), dtype="float32", always_2d=True)
                    v_1d = to_mono(xv)
                    v_codes = encode_1d_mono(v_1d, srv, model, fe, device=args.device, chunk_seconds=args.chunk_seconds)
                    v_codec_path = out_dir / f"{key}.Vocals.npy"
                    np.save(v_codec_path, v_codes)
                except Exception as e:
                    print(f"[{key}] WARN vocals: {e}")

            # --- encode INSTR stem (if present) ---
            i_codec_path = None
            if i_wav and i_wav.is_file():
                try:
                    xi, sri = sf.read(str(i_wav), dtype="float32", always_2d=True)
                    i_1d = to_mono(xi)
                    i_codes = encode_1d_mono(i_1d, sri, model, fe, device=args.device, chunk_seconds=args.chunk_seconds)
                    i_codec_path = out_dir / f"{key}.Instrumental.npy"
                    np.save(i_codec_path, i_codes)
                except Exception as e:
                    print(f"[{key}] WARN instrumental: {e}")

            # Fallbacks: if codec failed but instrumental exists, use instrumental as codec
            if codec_path is None and i_codec_path is not None:
                codec_path = i_codec_path

            entry = {
                "id": f"voidvoid_{i}",
                "codec": codec_path.as_posix() if codec_path else None,
                "vocals_codec": v_codec_path.as_posix() if v_codec_path else None,
                "instrumental_codec": i_codec_path.as_posix() if i_codec_path else None,
                "audio_length_in_sec": length_sec,
                "msa": [],
                "genres": args.tags,
                "splitted_lyrics": {"segmented_lyrics": []},
                "text": ""  # safe default for COT preprocessor
            }
            jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            made += 1

    print(f"Done. Wrote {made} entries to {jsonl_path}")
    print(f"Codes saved under {out_dir}")

if __name__ == "__main__":
    main()

