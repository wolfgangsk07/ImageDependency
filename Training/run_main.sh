#!/bin/bash



cancer_types=(
    PAAD PCPG
)


prediction_types=("tformer" "tformer_lin" "Agent" "MLP")


extraction_models=("vit" "resnet")


for cancer in "${cancer_types[@]}"; do
    for pred in "${prediction_types[@]}"; do
        for model in "${extraction_models[@]}"; do

            python /backup/lgx/path_omics_t/script/main.py \
                --cancer_type "$cancer" \
                --prediction_type "$pred" \
                --extraction_model "$model"
        done
    done
done


