# BERT 
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type bert \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type bert \
    --output_folder output2 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type bert \
    --output_folder output3 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 3


# ROBERTA
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type roberta \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type roberta \
    --output_folder output2 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type roberta \
    --output_folder output3 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 3


# ALBERT
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type albert \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type albert \
    --output_folder output2 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type albert \
    --output_folder output3 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 3


# ELECTRA 
CUDA_VISIBLE_DEVICES=5 python probe.py \
    --model_type t5 \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type electra \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type electra \
    --output_folder output2 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type electra \
    --output_folder output3 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 3


# GPT2
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type gpt2 \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type gpt2 \
    --output_folder output2 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type gpt2 \
    --output_folder output3 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 3


# T5
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type t5 \
    --output_folder output1 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 1 

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type t5 \
    --output_folder output2 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 2 

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type t5 \
    --output_folder output3 \
    --train_data data/data_en/train.tsv \
    --dev_data data/data_en/dev.tsv \
    --test_data data/data_en/test.tsv \
    --seed 3


# BERT-ZH
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-zh \
    --output_folder output1 \
    --train_data data/data_zh/train.tsv \
    --dev_data data/data_zh/dev.tsv \
    --test_data data/data_zh/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-zh \
    --output_folder output2 \
    --train_data data/data_zh/train.tsv \
    --dev_data data/data_zh/dev.tsv \
    --test_data data/data_zh/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-zh \
    --output_folder output3 \
    --train_data data/data_zh/train.tsv \
    --dev_data data/data_zh/dev.tsv \
    --test_data data/data_zh/test.tsv \
    --seed 3


# BERT-DE
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-de \
    --output_folder output1 \
    --train_data data/data_de/train.tsv \
    --dev_data data/data_de/dev.tsv \
    --test_data data/data_de/test.tsv \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-de \
    --output_folder output2 \
    --train_data data/data_de/train.tsv \
    --dev_data data/data_de/dev.tsv \
    --test_data data/data_de/test.tsv \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-de \
    --output_folder output3 \
    --train_data data/data_de/train.tsv \
    --dev_data data/data_de/dev.tsv \
    --test_data data/data_de/test.tsv \
    --seed 3
