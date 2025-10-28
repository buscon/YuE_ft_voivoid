import os
import json

npy_dir = "example/npy/"
jsonl_path = "example/jsonl/dummy.msa.xcodec_16k.jsonl"

npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")][:5]  # Use first 5 files

with open(jsonl_path, "w") as f:
    for i, npy_file in enumerate(npy_files, start=1):
        entry = {
            "id": str(i),
            "codec": f"example/npy/{npy_file}",
            "audio_length_in_sec": 85.16,
            "msa": [
                {"start": 0.0, "end": 13.93, "label": "intro"},
                {"start": 13.93, "end": 32.51, "label": "verse"},
                {"start": 32.51, "end": 51.09, "label": "chorus"},
                {"start": 51.09, "end": 85.17, "label": "outro"}
            ],
            "genres": "male, youth, powerful, charismatic, rock, punk",
            "splitted_lyrics": {
                "segmented_lyrics": [
                    {"offset": 0.0, "duration": 13.93, "codec_frame_start": 0, "codec_frame_end": 696, "line_content": "[intro]\n\n"},
                    {"offset": 13.93, "duration": 18.58, "codec_frame_start": 696, "codec_frame_end": 1625, "line_content": "[verse]\n\n"},
                    {"offset": 32.51, "duration": 18.58, "codec_frame_start": 1625, "codec_frame_end": 2554, "line_content": "[chorus]\n\n"},
                    {"offset": 51.09, "duration": 34.06, "codec_frame_start": 2554, "codec_frame_end": 4258, "line_content": "[outro]\n\n"}
                ]
            }
        }
        f.write(json.dumps(entry) + "\n")

