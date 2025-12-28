GPUS="4,5,6,7"

MODELS=(
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "google/medgemma-27b-text-it"
  "meta-llama/Llama-3.3-70B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  python3 ./syngen.py "$MODEL" "$GPUS"
  python3 ./synformat.py "$MODEL" "$GPUS"
done
