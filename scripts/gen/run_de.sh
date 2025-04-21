#!/bin/bash

set -ex

# export CUBLAS_WORKSPACE_CONFIG=:16:8  
# export CUDA_VISIBLE_DEVICES=1

BUDGET=10
POPSIZE=10
SEED=15
GA=topk

for dataset in dyck_languages gsm8k math multistep_arithmetic_two object_counting samsum word_sorting
do
OUT_PATH=outputs/gen/$dataset/de/bd${BUDGET}_top${POPSIZE}_para_topk_init/$GA/$LLM_TYPE
for SEED in 15
do
python3 run.py \
    --seed $SEED \
    --dataset $dataset \
    --metric meteor \
    --task gen \
    --language_model AnatoliiPotapov/T-lite-instruct-0.1 \
    --batch-size 32 \
    --prompt-num 0 \
    --sample_num 100 \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position demon \
    --evo_mode de \
    --setting default \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --cache_path data/gen/$dataset/seed${SEED}/prompts_batched.json \
    --output $OUT_PATH/seed$SEED \
    --dev_file ./data/gen/$dataset/seed${SEED}/dev.txt
done
done