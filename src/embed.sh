MODEL="google/medgemma-27b-text-it"

python3 ./embed.py "$MODEL" "0,1" "bigwiz83/sapbert-from-pubmedbert-squad2" 
python3 ./embed.py "$MODEL" "0,1" "emilyalsentzer/Bio_ClinicalBERT"
python3 ./embed.py "$MODEL" "0,1" "medicalai/ClinicalBERT"
python3 ./embed.py "$MODEL" "0,1" "dmis-lab/biobert-v1.1"
python3 ./embed.py "$MODEL" "0,1" "GanjinZero/UMLSBert_ENG"
python3 ./embed.py "$MODEL" "0,1" "allenai/scibert_scivocab_cased"