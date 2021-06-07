
#ZH

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-zh \
    --output_folder output1 \
    --train_data data/data_zh/train.json \
    --dev_data data/data_zh/dev.json \
    --test_data data/data_zh/test.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-zh \
    --output_folder output2 \
    --train_data data/data_zh/train.json \
    --dev_data data/data_zh/dev.json \
    --test_data data/data_zh/test.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-zh \
    --output_folder output3 \
    --train_data data/data_zh/train.json \
    --dev_data data/data_zh/dev.json \
    --test_data data/data_zh/test.json \
    --seed 3


# ES
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-es \
    --output_folder output1 \
    --train_data data/data_es/train.json \
    --dev_data data/data_es/dev.json \
    --test_data data/data_es/test.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-es \
    --output_folder output2 \
    --train_data data/data_es/train.json \
    --dev_data data/data_es/dev.json \
    --test_data data/data_es/test.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-es \
    --output_folder output3 \
    --train_data data/data_es/train.json \
    --dev_data data/data_es/dev.json \
    --test_data data/data_es/test.json \
    --seed 3


# DE
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-de \
    --output_folder output1 \
    --train_data data/data_de/train.json \
    --dev_data data/data_de/dev.json \
    --test_data data/data_de/test.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-de \
    --output_folder output2 \
    --train_data data/data_de/train.json \
    --dev_data data/data_de/dev.json \
    --test_data data/data_de/test.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bert-de \
    --output_folder output3 \
    --train_data data/data_de/train.json \
    --dev_data data/data_de/dev.json \
    --test_data data/data_de/test.json \
    --seed 3

