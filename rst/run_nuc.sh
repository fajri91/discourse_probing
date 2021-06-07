# BERT 
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type bert \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type bert \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type bert \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# ROBERTA
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type roberta \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type roberta \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type roberta \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# ALBERT
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type albert \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type albert \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --task nuclearity \
    --model_type albert \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# ELECTRA
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --task nuclearity \
    --model_type electra \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --task nuclearity \
    --model_type electra \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --task nuclearity \
    --model_type electra \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# GPT2
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --task nuclearity \
    --model_type gpt2 \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --task nuclearity \
    --model_type gpt2 \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --task nuclearity \
    --model_type gpt2 \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# BART
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bart \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bart \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bart \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# T5
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type t5 \
    --output_folder output1 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 1 

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type t5 \
    --output_folder output2 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 2  

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type t5 \
    --output_folder output3 \
    --train_data data/data_en/train.csv \
    --dev_data data/data_en/dev.csv \
    --test_data data/data_en/test.csv \
    --seed 3


# BERT-ZH
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-zh \
    --output_folder output1 \
    --train_data data/data_zh/train.csv \
    --dev_data data/data_zh/dev.csv \
    --test_data data/data_zh/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-zh \
    --output_folder output2 \
    --train_data data/data_zh/train.csv \
    --dev_data data/data_zh/dev.csv \
    --test_data data/data_zh/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-zh \
    --output_folder output3 \
    --train_data data/data_zh/train.csv \
    --dev_data data/data_zh/dev.csv \
    --test_data data/data_zh/test.csv \
    --seed 3


# BERT-ES
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-es \
    --output_folder output1 \
    --train_data data/data_es/train.csv \
    --dev_data data/data_es/dev.csv \
    --test_data data/data_es/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-es \
    --output_folder output2 \
    --train_data data/data_es/train.csv \
    --dev_data data/data_es/dev.csv \
    --test_data data/data_es/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-es \
    --output_folder output3 \
    --train_data data/data_es/train.csv \
    --dev_data data/data_es/dev.csv \
    --test_data data/data_es/test.csv \
    --seed 3


# BERT-DE
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-de \
    --output_folder output1 \
    --train_data data/data_de/train.csv \
    --dev_data data/data_de/dev.csv \
    --test_data data/data_de/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-de \
    --output_folder output2 \
    --train_data data/data_de/train.csv \
    --dev_data data/data_de/dev.csv \
    --test_data data/data_de/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --task nuclearity \
    --model_type bert-de \
    --output_folder output3 \
    --train_data data/data_de/train.csv \
    --dev_data data/data_de/dev.csv \
    --test_data data/data_de/test.csv \
    --seed 3
