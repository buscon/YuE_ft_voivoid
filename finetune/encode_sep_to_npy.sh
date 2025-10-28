# encode_sep_to_npy.sh
#!/usr/bin/env bash
set -euo pipefail

SEP_DIR="${1:-example/sep}"        # your separated WAVs folder
NPY_DIR="${2:-example/npy}"        # where .npy codes will be written

# Change this to your actual xcodec encoder CLI
XCODEC="${XCODEC:-xcodec_cli}"     # e.g., /path/to/xcodec_cli

echo "Encoding stems from $SEP_DIR → $NPY_DIR"
find "$SEP_DIR" -type f -name "*.wav" -print0 | while IFS= read -r -d '' wav; do
  rel="${wav#"$SEP_DIR"/}"
  out="$NPY_DIR/${rel%.wav}.npy"
  mkdir -p "$(dirname "$out")"
  # Example invocation; replace flags to match your encoder:
  "$XCODEC" encode --sr 44100 --input "$wav" --output "$out"
  echo "✓ $(basename "$wav") → $(basename "$out")"
done

echo "Done."

