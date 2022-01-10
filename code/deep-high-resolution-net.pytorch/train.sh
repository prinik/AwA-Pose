#/bin/bash
#conda activate hand
#module load gcc/8.1.0
#CUDA_VISIBLE_DEVICES=7 python tools/train.py --cfg experiments/atrw/w48_384x288.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/awa/w48_384x288.yaml
