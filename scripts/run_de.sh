#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=10
POPSIZE=10
SEED=15
GA=topk

for dataset in sst-2
do
OUT_PATH=outputs/cls/$dataset/alpaca/all/ga/bd${BUDGET}_top${POPSIZE}_para_topk_init/$GA/$LLM_TYPE
for SEED in 15
do
CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
    --seed $SEED \
    --dataset $dataset \
    --task cls \
    --language_model AnatoliiPotapov/T-lite-instruct-0.1 \
    --batch-size 32 \
    --prompt-num 0 \
    --sample_num 500 \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position demon \
    --evo_mode de \
    --setting default \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --cache_path data/cls/$dataset/seed${SEED}/prompts_batched.json \
    --output $OUT_PATH/seed$SEED \
    --dev_file ./data/cls/$dataset/seed${SEED}/dev.txt
done
python3 get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done