GPUS="5,6"

MODELS=(
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "Qwen/Qwen3-4B-Instruct-2507"
  "google/medgemma-27b-text-it"
  "google/medgemma-4b-it"
)

python3 "transform.py"

for MODEL in "${MODELS[@]}"; do
  python3 ./synclass.py "$MODEL" "$GPUS"
done

for MODEL in "${MODELS[@]}"; do
  python3 ./synclassformat.py "$MODEL" "$GPUS"
done

python3 "./synclassmerge.py"
python3 "./synclasseval.py"
