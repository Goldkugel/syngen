GPUS="4,5,6,7"

MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen3-4B-Instruct-2507"
  "google/medgemma-4b-it"
  "meta-llama/Llama-3.1-8B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  python3 ./syngen.py "$MODEL" "$GPUS"
  python3 ./synformat.py "$MODEL" "$GPUS"
done
