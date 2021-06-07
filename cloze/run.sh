# BERT
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert \
    --output_folder output1 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert \
    --output_folder output2 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert \
    --output_folder output3 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 3


# ROBERTA
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type roberta \
    --output_folder output1 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type roberta \
    --output_folder output2 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type roberta \
    --output_folder output3 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 3


# ALBERT
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type albert \
    --output_folder output1 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type albert \
    --output_folder output2 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type albert \
    --output_folder output3 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 3


# ELECTRA
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type electra \
    --output_folder output1 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type electra \
    --output_folder output2 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type electra \
    --output_folder output3 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 3


# GPT2
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type gpt2 \
    --output_folder output1 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type gpt2 \
    --output_folder output2 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type gpt2 \
    --output_folder output3 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 3


# T5
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type t5 \
    --output_folder output1 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 1 

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type t5 \
    --output_folder output2 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 2 

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type t5 \
    --output_folder output3 \
    --train_data data/train.csv \
    --dev_data data/dev.csv \
    --test_data data/test.csv \
    --seed 3 

