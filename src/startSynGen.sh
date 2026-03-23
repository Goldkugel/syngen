GPUS="4,5,6,7"

MODELS=(
  # Execution time: 19 hours, 58 minutes, and 58 seconds 
  #"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  # Execution time: 1 day, 19 hours, 59 minutes, and 44 seconds
  #"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  # Execution time: 1 day, 23 hours, 42 minutes, and 44 seconds 
  "google/medgemma-27b-text-it"
)

./prepare.sh

for MODEL in "${MODELS[@]}"; do
  python3 ./syngen.py "$MODEL" "$GPUS"
done

for MODEL in "${MODELS[@]}"; do
  python3 ./synformat.py "$MODEL" "$GPUS"
done

python3 "./merge.py"
python3 "./eval.py"
