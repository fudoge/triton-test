#! /usr/bin/env bash

set -e

if [ $# -ne 1 ]; then
    echo "usage: ./request_test.sh <endpoint>"
    exit 0
fi

MODEL_ENDPOINT=$1

(
cat infer.json
cat smiles_input.bin
cat fasta_input.bin
) | curl -s -X POST ${MODEL_ENDPOINT} \
-H "Content-Type: application/octet-stream" \
-H "Inference-Header-Content-Length: $(wc -c < infer.json)" \
-H "Expect:" \
--data-binary @- | jq .
