GPUS="0,1"

# python3 ./synclass.py "google/medgemma-27b-text-it" "0,1"

MODELS=(
  #"google/medgemma-4b-it"
  #"Qwen/Qwen3-4B-Instruct-2507"
  #"mistralai/Mistral-7B-Instruct-v0.2"
  #"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  #"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "google/medgemma-27b-text-it"
)

./prepare.sh
./embed.sh

for MODEL in "${MODELS[@]}"; do
  python3 ./synclass.py "$MODEL" "$GPUS"
done

for MODEL in "${MODELS[@]}"; do
  python3 ./synclassformat.py "$MODEL" "$GPUS"
done

python3 "./synclassmerge.py"
python3 "./synclasseval.py"
