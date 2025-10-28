from encodec import EncodecModel
import torch
import torchaudio
import os
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = EncodecModel.encodec_model_24khz().to(device)
model.set_target_bandwidth(6.0)

input_dir = "voivoid_dataset/wav/"
output_dir = "voivoid_dataset/npy/"
os.makedirs(output_dir, exist_ok=True)

for wav in os.listdir(input_dir):
    if wav.endswith(".wav"):
        input_path = os.path.join(input_dir, wav)
        print(f"Processing: {wav}")

        try:
            # Load and preprocess
            wav_data, sr = torchaudio.load(input_path)
            if sr != 24000:
                transform = torchaudio.transforms.Resample(sr, 24000)
                wav_data = transform(wav_data)
            if wav_data.shape[0] == 2:
                wav_data = wav_data.mean(dim=0, keepdim=True)
            wav_data = wav_data.to(device).contiguous()

            # Encode
            with torch.no_grad():
                result = model.encode(wav_data.unsqueeze(0))
                codes = result[0][0]

            # Save
            output_path = os.path.join(output_dir, wav.replace(".wav", ".npy"))
            torch.save(codes.cpu().numpy(), output_path)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Skipped {wav} due to error: {e}")
            continue

        # Free memory
        del wav_data, result, codes
        gc.collect()
        torch.cuda.empty_cache()

