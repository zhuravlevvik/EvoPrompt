#!/bin/bash

set -ex

# export CUBLAS_WORKSPACE_CONFIG=:16:8  
# export CUDA_VISIBLE_DEVICES=1

BUDGET=10
POPSIZE=10
SEED=15
GA=topk

for dataset in boolean_expressions causal_judgement date_understanding disambiguation_qa formal_fallacies geometric_shapes hyperbaton logical_deduction_five_objects logical_deduction_three_objects logical_deduction_seven_objects movie_recommendation navigate penguins_in_a_table reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks sports_understanding temporal_sequences tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects
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