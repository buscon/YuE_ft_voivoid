from encodec import EncodecModel
import torch
import torchaudio
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = EncodecModel.encodec_model_24khz().to(device)
model.set_target_bandwidth(6.0)

input_path = "voivoid_dataset/wav/Voivod_-_The_Outer_Limits_(1993)_[16B-44.1kHz]_07. Jack Luminous.wav"
wav_data, sr = torchaudio.load(input_path)
print(f"Shape: {wav_data.shape}, Sample rate: {sr}, Dtype: {wav_data.dtype}, Is contiguous: {wav_data.is_contiguous()}")

