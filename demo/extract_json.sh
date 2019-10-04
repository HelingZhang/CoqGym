#!/bin/bash
META_FILE_PATH='./demo/scratch.meta'
JSON_FILE_PATH='./data/demo/scratch.json'
PROOF_NAMES_FILE_PATH="proof_names.txt"

cd ..
# check proofs in scratch.v
python check_proofs.py \
  --file $META_FILE_PATH

# extract the names of every proof in scratch.v
python demo/extract_proof_names.py \
  --file $JSON_FILE_PATH

# for each proof in scratch.v name run ectract_proof.py
while IFS= read -r PROOF_NAME
do
    echo "$PROOF_NAME"
    python extract_proof.py \
      --file $META_FILE_PATH \
      --proof $PROOF_NAME
done < "$PROOF_NAMES_FILE_PATH"

# post processing, merging json data of all proofs into one scratch.json
python postprocess.py

cd demo