gpu=0
ep=100
lr=0.5
bs=256
iter=5
ss=2
eps=8
beta=8  # reported hparam for ProFeAT WRN34w10-c100
wd=3e-4  # reported hparam for ProFeAT WRN34w10-c100
lin_lr=5.0
ROOT="path/to/dataset"

proj_hdim=640
proj_odim=256

name="this_run_description"

CUDA_VISIBLE_DEVICES=$gpu python pretraining.py \
--arch WideResNet34 \
--dataset cifar100 \
--root $ROOT \
--epoch $ep \
--batch_size $bs \
--lr $lr \
--weight_decay $wd \
--epsilon $eps \
--iter $iter \
--step_size $ss \
--beta $beta \
--proj_hdim $proj_hdim \
--proj_odim $proj_odim \
--PT_ckpt path/to/ssl/teacher/checkpoint \
-m $name \
--linear_eval \
--linear_lr $lin_lr \
--entity my_entity \
--project my_project
