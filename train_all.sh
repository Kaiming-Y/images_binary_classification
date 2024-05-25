#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <epoch>"
  exit 1
fi

EPOCH=$1

cd ./src || { echo "Failed to change directory to ./src"; exit 1; }

LOG_FILE="../training.log"
models=('googlenet' 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')

> $LOG_FILE

# Train every model
for model in "${models[@]}"
do
    echo "Training model: $model" | tee -a $LOG_FILE
    
    python train.py --model $model --epoch $EPOCH 2>&1 | tee -a $LOG_FILE   
    python train.py --model $model --augment --epoch $EPOCH 2>&1 | tee -a $LOG_FILE
    
    echo "Completed training model: $model" | tee -a $LOG_FILE
done

echo "All models have been trained." | tee -a $LOG_FILE
