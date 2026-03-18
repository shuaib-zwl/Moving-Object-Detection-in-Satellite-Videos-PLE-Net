#!/usr/bin/env bash

# 必选参数
CONFIG=$1
CHECKPOINT=$2
GPUS=$3

# 可选环境变量
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}

# 把项目根目录加入 PYTHONPATH（确保 mmcv/mmdet 能被 import）
export PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH

# 分布式启动
torchrun --nnodes=$NNODES \
         --nproc_per_node=$GPUS \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
