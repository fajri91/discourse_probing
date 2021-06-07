# BERT
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3


# ROBERTA
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type roberta \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type roberta \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type roberta \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3


# ALBERT
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type albert \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type albert \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type albert \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3



# ELECTRA
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type electra \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type electra \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type electra \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3


# GPT2
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type gpt2 \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type gpt2 \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type gpt2 \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3


# BART
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bart \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bart \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bart \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3


# T5
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type t5 \
    --output_folder output1 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type t5 \
    --output_folder output2 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type t5 \
    --output_folder output3 \
    --train_data data/data_en/train_edu.json \
    --dev_data data/data_en/dev_edu.json \
    --test_data data/data_en/test_edu.json \
    --seed 3


# BERT-ZH
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type bert-zh \
    --output_folder output1 \
    --train_data data/data_zh/train_edu.json \
    --dev_data data/data_zh/dev_edu.json \
    --test_data data/data_zh/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type bert-zh \
    --output_folder output2 \
    --train_data data/data_zh/train_edu.json \
    --dev_data data/data_zh/dev_edu.json \
    --test_data data/data_zh/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type bert-zh \
    --output_folder output3 \
    --train_data data/data_zh/train_edu.json \
    --dev_data data/data_zh/dev_edu.json \
    --test_data data/data_zh/test_edu.json \
    --seed 3


# BERT-DE
CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type bert-de \
    --output_folder output1 \
    --train_data data/data_de/train_edu.json \
    --dev_data data/data_de/dev_edu.json \
    --test_data data/data_de/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type bert-de \
    --output_folder output2 \
    --train_data data/data_de/train_edu.json \
    --dev_data data/data_de/dev_edu.json \
    --test_data data/data_de/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=7 python probe.py \
    --model_type bert-de \
    --output_folder output3 \
    --train_data data/data_de/train_edu.json \
    --dev_data data/data_de/dev_edu.json \
    --test_data data/data_de/test_edu.json \
    --seed 3


# BERT-ES
CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-es \
    --output_folder output1 \
    --train_data data/data_es/train_edu.json \
    --dev_data data/data_es/dev_edu.json \
    --test_data data/data_es/test_edu.json \
    --seed 1

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-es \
    --output_folder output2 \
    --train_data data/data_es/train_edu.json \
    --dev_data data/data_es/dev_edu.json \
    --test_data data/data_es/test_edu.json \
    --seed 2

CUDA_VISIBLE_DEVICES=1 python probe.py \
    --model_type bert-es \
    --output_folder output3 \
    --train_data data/data_es/train_edu.json \
    --dev_data data/data_es/dev_edu.json \
    --test_data data/data_es/test_edu.json \
    --seed 3
