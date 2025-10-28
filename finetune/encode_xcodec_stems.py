#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import XcodecModel, AutoFeatureExtractor

def find_pairs(root: Path):
    vocals, inst = {}, {}
    for p in root.rglob("*.wav"):
        n = p.name
        if n.endswith(".Vocals.wav"):
            key = n[:-len(".Vocals.wav")]
            vocals[key] = p
        elif n.endswith(".Instrumental.wav"):
            key = n[:-len(".Instrumental.wav")]
            inst[key] = p
    keys = sorted(set(vocals) | set(inst))
    return [(k, vocals.get(k), inst.get(k)) for k in keys]

def load_audio(path: Path, target_sr: int):
    import torchaudio
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    # resample if needed
    if sr != target_sr:
        wav_t = torch.from_numpy(wav.T)  # (C, T)
        wav_t = torchaudio.transforms.Resample(sr, target_sr)(wav_t)
        wav = wav_t.T.numpy()
        sr = target_sr
    # downmix to mono → 1-D (T,)
    if wav.shape[1] > 1:
        wav = wav.mean(axis=1)
    else:
        wav = wav[:, 0]
    return wav, sr  # (T,)

def encode_wav_to_codes(wav_1d: np.ndarray, sr: int, model, fe, device="cuda"):
    """
    wav_1d: (T,) float32 in [-1,1]
    DAC/X-Codec FE expects mono 1-D. Model wants (B, C, T) → we’ll add dims.
    """
    # Feature extractor on 1-D mono
    inputs = fe(raw_audio=wav_1d, sampling_rate=sr, return_tensors="pt")
    iv = inputs["input_values"]  # usually shape (T,) or (1, T)

    # Ensure (B, C, T)
    if iv.ndim == 1:           # (T,) → (1,1,T)
        iv = iv.unsqueeze(0).unsqueeze(0)
    elif iv.ndim == 2:         # (1, T) or (C, T) with C==1 → (1,1,T)
        if iv.shape[0] == 1:
            iv = iv.unsqueeze(1)
        else:
            # if extractor ever returns (C,T) with C>1 (shouldn't for mono), keep first channel
            iv = iv[:1, :].unsqueeze(0)
    # else: if already (B,C,T) do nothing

    iv = iv.to(device)
    with torch.no_grad():
        out = model.encode(iv)  # audio_codes: (B, Q, L)
        codes = out.audio_codes[0].cpu().numpy().astype(np.int64)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="example/sep")
    ap.add_argument("--out_dir", default="example/npy")
    ap.add_argument("--jsonl", default="example/jsonl/voidvoid_sep.jsonl")
    ap.add_argument("--model_id", default="hf-audio/xcodec-hubert-librispeech")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tags", default="male, youth, powerful, charismatic, rock, punk")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl).resolve(); jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    model = XcodecModel.from_pretrained(args.model_id).to(args.device)
    fe = AutoFeatureExtractor.from_pretrained(args.model_id)
    target_sr = fe.sampling_rate

    pairs = find_pairs(in_dir)
    total_stems = 0
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, (key, v_wav, i_wav) in enumerate(pairs, 1):
            entry = {
                "id": f"voidvoid_{i}",
                "codec": None,
                "vocals_codec": None,
                "instrumental_codec": None,
                "audio_length_in_sec": None,
                "msa": [],
                "genres": args.tags,
                "splitted_lyrics": {"segmented_lyrics": []}
            }
            # use whichever stem exists to set duration
            dur_set = False
            for kind, wav_path in (("Vocals", v_wav), ("Instrumental", i_wav)):
                if wav_path is None: continue
                wav, sr = load_audio(wav_path, target_sr)
                codes = encode_wav_to_codes(wav, sr, model, fe, device=args.device)
                out_npy = out_dir / f"{key}.{kind}.npy"
                np.save(out_npy, codes)
                if not dur_set:
                    entry["audio_length_in_sec"] = float(wav.shape[0] / sr)
                    dur_set = True
                if kind == "Vocals":
                    entry["vocals_codec"] = out_npy.as_posix()
                else:
                    entry["instrumental_codec"] = out_npy.as_posix()
                total_stems += 1
            jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Encoded {total_stems} stems. JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()

