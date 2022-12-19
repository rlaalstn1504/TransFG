# TransFG: A Transformer Architecture for Fine-grained Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/image-classification-on-inaturalist)](https://paperswithcode.com/sota/image-classification-on-inaturalist?p=transfg-a-transformer-architecture-for-fine)

Official PyTorch code for the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  

![](./TransFG.png)

## Dependencies:
+ Python 3.7.3
+ PyTorch 1.5.1
+ torchvision 1.8.2+cu111
+ ml_collections

### 1. packages 설치

Install dependencies with the following command:

```bash
pip3 install -r requirements.txt
```

### 2. 데이터 셋 준비
##### 2.1 make_data 폴더에 raw_data 준비(라벨폴더, 이미지폴더)
##### 2.2 make_data 폴더의 make_image_folder.ipynb 파일을 실행하여 datasets/custom 폴더에 폴더(라벨명)/이미지 형태로 이미지 생성
##### 2.3 make_data 폴더의 text_to_df.ipynb 파일을 실행하여 train/val/test 데이터 경로가 담긴 csv 파일 생성  

### 3. Train

To train TransFG on custom dataset with 1 gpus in FP-16 mode for 10000 steps run:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py --dataset custum --split overlap --num_steps 10000 --fp16 --name sample_run
```
To train TransFG on custom dataset with 4 gpus in FP-16 mode for 10000 steps run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset custum --split overlap --num_steps 10000 --fp16 --name sample_run
```
