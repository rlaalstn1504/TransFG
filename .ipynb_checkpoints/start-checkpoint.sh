#!/bin/bash
echo date
date
echo ## start of training
echo CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py --dataset custom --split overlap --fp16 --name sample_run
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py --dataset custom --split overlap --fp16 --name sample_run 
echo ## end of training
echo date
date
