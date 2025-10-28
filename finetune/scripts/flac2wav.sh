find ~/Qobuz\ Downloads/ -type f -name "*.flac" -exec bash -c '
  for flac; do
    album=$(basename "$(dirname "$flac")" | sed "s/ /_/g")
    track=$(basename "$flac" .flac)
    ffmpeg -i "$flac" -ar 44100 -ac 2 -c:a pcm_s16le ~/Documents/YuE_ft_voivoid/finetune/voivoid_dataset/wav/"${album}_${track}.wav"
  done' bash {} +

