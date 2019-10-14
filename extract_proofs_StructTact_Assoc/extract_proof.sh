#!/bin/bash
input="proofnames_StructTact_Assoc.txt"
while IFS= read -r line
do
    echo "extracting $line"
    python extract_proof.py --file ../coq_projects/StructTact/Assoc.meta --proof $line
done < "$input"
