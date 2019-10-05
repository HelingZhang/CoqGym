#!/bin/bash
META_FILE_PATH='./demo/scratch.meta'
JSON_FILE_PATH='./data/demo/scratch.json'
PROOF_NAMES_FILE_PATH="proof_names.txt"

cd ..
# check proofs in scratch.v
echo -e '\e[1m\e[34m Step (1/4) -- check proofs \e[39m\e[0m'
python check_proofs.py \
  --file $META_FILE_PATH

# extract the names of every proof in scratch.v
echo -e '\e[1m\e[34m Step (2/4) -- extract proof names \e[39m\e[0m'
python demo/extract_proof_names.py \
  --file $JSON_FILE_PATH \
  --save_path $PROOF_NAMES_FILE_PATH

# for each proof in scratch.v name run ectract_proof.py
echo -e '\e[1m\e[34m Step (3/4) -- extract proofs by proof name \e[39m\e[0m'
while IFS= read -r PROOF_NAME
do
    echo "$PROOF_NAME"
    python extract_proof.py \
      --file $META_FILE_PATH \
      --proof $PROOF_NAME
done < "$PROOF_NAMES_FILE_PATH"

# post processing, merging json data of all proofs into one scratch.json
echo -e '\e[1m\e[34m Step (3/4) -- postprocessing \e[39m\e[0m'
python postprocess.py

cd demo