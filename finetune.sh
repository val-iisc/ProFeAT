ROOT="path/to/dataset"

gpu=0
dataset="cifar100"
lr=5.0

name="my_run_name"

CUDA_VISIBLE_DEVICES=$gpu python finetuning.py \
--arch WideResNet34 \
--epochs 25 \
--dataset $dataset \
--root $ROOT \
--lr $lr \
--weight_decay 2e-4 \
--batch_size 512 \
--test_batch_size 256 \
--fixmode f3 \
--fixbn \
--trainmode normal \
--offline \
--name $name \
--checkpoint path/to/sslat/checkpoint