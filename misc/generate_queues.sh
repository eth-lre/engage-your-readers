#!/usr/bin/bash

SRC_PREFIX="/home/vilda/Downloads/Sustainability Business and the Environment"
TGT_PREFIX="annotation_ui/web/queues/"

for i in $(seq 10 19); do
    cp "${SRC_PREFIX}_control.json" "${TGT_PREFIX}/u${i}_quant.jsonl"
done

for i in $(seq 20 29); do
    cp "${SRC_PREFIX}_authentic.json" "${TGT_PREFIX}/u${i}_pogos.jsonl"
done

for i in $(seq 30 39); do
    cp "${SRC_PREFIX}_generated.json" "${TGT_PREFIX}/u${i}_jacon.jsonl"
done