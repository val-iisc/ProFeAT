gpu=0
dataset="cifar100"
ROOT="path/to/dataset"
ckpt_path="path/to/sslat/checkpoint"
arch="WideResNet34"

CUDA_VISIBLE_DEVICES=$gpu python autoattack_eval.py \
$ckpt_path \
--arch $arch \
--dataset $dataset \
--root $ROOT \
--entity my_entity \
--project my_project