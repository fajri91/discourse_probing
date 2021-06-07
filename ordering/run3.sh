
#BART MODEL

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bart \
    --output_folder output1 \
    --train_data data/data_en/train.json \
    --dev_data data/data_en/dev.json \
    --test_data data/data_en/test.json \
    --seed 1 
 
CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bart \
    --output_folder output2 \
    --train_data data/data_en/train.json \
    --dev_data data/data_en/dev.json \
    --test_data data/data_en/test.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type bart \
    --output_folder output3 \
    --train_data data/data_en/train.json \
    --dev_data data/data_en/dev.json \
    --test_data data/data_en/test.json \
    --seed 3


# t5

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type t5 \
    --output_folder output1 \
    --train_data data/data_en/train.json \
    --dev_data data/data_en/dev.json \
    --test_data data/data_en/test.json \
    --seed 1

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type t5 \
    --output_folder output2 \
    --train_data data/data_en/train.json \
    --dev_data data/data_en/dev.json \
    --test_data data/data_en/test.json \
    --seed 2

CUDA_VISIBLE_DEVICES=0 python probe.py \
    --model_type t5 \
    --output_folder output3 \
    --train_data data/data_en/train.json \
    --dev_data data/data_en/dev.json \
    --test_data data/data_en/test.json \
    --seed 3
