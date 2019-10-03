#!/bin/bash
FILE_PATH='./demo/scratch.meta'
PROOF_NAME='mult_S_1'

cd /home/heling/fun/CoqGym/demo
mkdir data

cd ..
python check_proofs.py \
  --file $FILE_PATH
python extract_proof.py \
  --file $FILE_PATH \
  --proof $PROOF_NAME
python postprocess.py
cd ./demo
